# -*- coding: utf-8 -*-
# Para executar esta aplicação:
# 1. Crie um arquivo 'requirements.txt' com o conteúdo abaixo.
# 2. Instale as bibliotecas: pip install -r requirements.txt
# 3. Execute no terminal: streamlit run app.py

import streamlit as st
import time
from datetime import datetime, timedelta
import pytz
import pandas as pd
from threading import Thread, Event
from queue import Queue
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import re
from scipy import stats
import base64
from io import BytesIO

# --- NOVAS IMPORTAÇÕES PARA GERAÇÃO DE PDF ---
from weasyprint import HTML

from c8y_api import CumulocityApi
from c8y_api.model import Alarm, Event as C8yEvent

# --- Adicionado para a refatoração com Dataclasses ---
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- NOVA IMPORTAÇÃO PARA O LIMIAR INTELIGENTE ---
from sklearn.cluster import KMeans


# --- Estruturas de Dados (Dataclasses) ---
@dataclass
class ConnectionConfig:
    """Configurações de conexão com a plataforma Cumulocity."""
    tenant_url: str
    username: str
    password: str


@dataclass
class DeviceAnalysisConfig:
    """Parâmetros de análise específicos para um único dispositivo."""
    device_id: str
    device_display_name: str
    target_measurements_list: List[str]
    is_mkpred: bool
    load_measurement_names: List[str] = field(default_factory=list)
    operating_current: float = 0.0
    stabilization_delay: int = 0
    shutdown_delay: int = 0
    startup_duration: int = 0
    operation_filter_mode: str = 'none'
    manual_thresholds: Dict[str, float] = field(default_factory=dict)
    # --- MKPRED KPI CONFIGS ---
    measurement_limits: Dict[str, float] = field(default_factory=dict)
    health_kpi_weights: Dict[str, float] = field(default_factory=lambda: {'severity': 0.4, 'degradation': 0.6})
    # --- REFRIGERATION KPI CONFIGS ---
    refrigeration_limit_mode: str = 'manual'
    refrigeration_kpi_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    refrigeration_kpi_weights: Dict[str, float] = field(
        default_factory=lambda: {'availability': 0.5, 'stability': 0.3, 'performance': 0.2})
    acceptable_variation_percent: float = 10.0


@dataclass
class AnalysisJob:
    """Define um trabalho de análise completo a ser executado."""
    connection: ConnectionConfig
    device_config: DeviceAnalysisConfig
    date_from: str
    date_to: str
    job_label: str
    fetch_alarms: bool
    fetch_events: bool


# --- Configuração da Página ---
st.set_page_config(
    page_title="Analisador de Performance de Ativos",
    page_icon="🔧",
    layout="wide"
)

# --- Estilo CSS Personalizado ---
st.markdown("""
<style>
    /* Esconde a barra lateral no modo de relatório */
    .report-mode [data-testid="stSidebar"] {
        display: none;
    }
    .log-container {
        background-color: #1a1a1a;
        color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 0.875rem;
        border: 1px solid #2D3748;
    }
    .log-entry { margin-bottom: 0.25rem; }
    .log-error { color: #ff4b4b; }
    .log-warning { color: #ffc400; }
    .log-debug { color: #808080; }
    .log-success { color: #28a745; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        background-color: #262730;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1a1a1a;
    }
    /* Estilo para os cartões de métrica */
    div[data-testid="metric-container"] {
        background-color: #262730;
        border: 1px solid #2D3748;
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    /* Estilo para o modo de impressão/relatório */
    @media print {
        .no-print {
            display: none !important;
        }
    }
</style>
""", unsafe_allow_html=True)


# --- Funções Auxiliares ---
def format_timestamp_to_brasilia(dt_obj):
    if not dt_obj or pd.isna(dt_obj): return ""
    try:
        if isinstance(dt_obj, str):
            dt_obj = datetime.fromisoformat(dt_obj.replace("Z", "+00:00"))
        brasilia_tz = pytz.timezone("America/Sao_Paulo")
        if dt_obj.tzinfo is None:
            dt_obj = pytz.utc.localize(dt_obj)
        return dt_obj.astimezone(brasilia_tz).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(dt_obj)


def format_uptime(total_seconds):
    if pd.isna(total_seconds) or total_seconds < 0:
        return "N/A"

    total_seconds = int(total_seconds)
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")

    if not parts and seconds >= 0:
        return f"{seconds}s"

    return " ".join(parts) if parts else "0s"


def extract_measurement_value(measurement, measurement_type):
    if measurement_type not in measurement:
        for key in measurement.keys():
            if key.startswith(measurement_type):
                try:
                    fragment = measurement[key]
                    first_series = next(iter(fragment.values()))
                    return float(first_series['value'])
                except (StopIteration, KeyError, ValueError, TypeError):
                    continue
        return None
    try:
        fragment = measurement[measurement_type]
        first_series = next(iter(fragment.values()))
        return float(first_series['value'])
    except (StopIteration, KeyError, ValueError, TypeError):
        return None


def calculate_health_index(kpis, device_config: DeviceAnalysisConfig):
    """Calcula o Índice de Saúde para um dispositivo de refrigeração com limites e pesos configuráveis."""
    if not kpis or kpis.get('is_mkpred'):
        return 0

    weights = device_config.refrigeration_kpi_weights
    limits = device_config.refrigeration_kpi_limits

    # --- Componente de Disponibilidade ---
    availability_score = kpis.get('availability', 100)

    # --- Componente de Estabilidade ---
    number_of_faults = kpis.get('number_of_faults', 0)
    stability_score = max(0, 100 - (number_of_faults * 10))  # Penalidade de 10 pontos por falha

    # --- Componente de Performance (Média + Estabilidade) ---
    performance_scores = []
    mean_values = kpis.get('mean_values', {})
    std_dev_values = kpis.get('std_dev_values', {})

    for param, limit_values in limits.items():
        mean_val = mean_values.get(param)
        std_dev_val = std_dev_values.get(param)

        if mean_val is not None and std_dev_val is not None:
            # 1. Nota da Média
            optimal_min = limit_values.get('min', 0)
            optimal_max = limit_values.get('max', 0)

            if optimal_min <= mean_val <= optimal_max:
                mean_score = 100
            else:
                distance = min(abs(mean_val - optimal_min), abs(mean_val - optimal_max))
                range_span = optimal_max - optimal_min
                if range_span > 0:
                    penalty = (distance / range_span) * 100
                    mean_score = max(0, 100 - penalty)
                else:
                    mean_score = 0

            # 2. Nota de Estabilidade
            range_span = optimal_max - optimal_min
            if range_span > 0:
                max_allowed_std_dev = range_span * (device_config.acceptable_variation_percent / 100.0)
                if std_dev_val <= max_allowed_std_dev and max_allowed_std_dev > 0:
                    stability_param_score = 100 - (std_dev_val / max_allowed_std_dev) * 100
                else:
                    stability_param_score = 0
            else:
                stability_param_score = 100  # Se não há faixa, não penaliza a estabilidade

            # A nota de performance do parâmetro é a média das duas notas
            combined_param_score = (mean_score + stability_param_score) / 2
            performance_scores.append(combined_param_score)

    if not performance_scores:
        final_performance_score = 100
    else:
        final_performance_score = np.mean(performance_scores)

    # --- Cálculo Final Ponderado ---
    health_index = (availability_score * weights['availability']) + \
                   (stability_score * weights['stability']) + \
                   (final_performance_score * weights['performance'])

    return max(0, min(100, health_index))


def is_likely_mkpred(series_list):
    """Heurística para verificar se um dispositivo é para análise de vibração (MKPRED)."""
    vibration_pattern = re.compile(r'^S\d+_(AC|VEL)_\d+$')
    legacy_vibration_measurements = {'v_rms', 'a_rms', 'a_peak', 'kurtosis', 'crest_factor', 'temperature'}
    motor_measurements = {'MA_01', 'MA_02'}

    has_vibration = any(
        vibration_pattern.match(s.split('.')[0]) or s.split('.')[0] in legacy_vibration_measurements
        for s in series_list
    )
    has_motor = any(s.split('.')[0] in motor_measurements for s in series_list)

    return has_vibration and not has_motor


# --- Funções de API (Thread-safe) ---
@st.cache_data(ttl=300)
def fetch_devices(tenant, user, password):
    try:
        full_tenant_id = tenant.split('.')[0].split('//')[1]
        if '-' in full_tenant_id:
            formatted_user = f"{full_tenant_id}/{user}"
        else:
            formatted_user = user
        c8y = CumulocityApi(base_url=tenant, tenant_id=full_tenant_id, username=formatted_user, password=password)
        all_devices = c8y.inventory.select(query="$filter=has(c8y_IsDevice)")

        devices_structured_list = []
        for device in all_devices:
            name = device.name or "Dispositivo sem nome"
            try:
                serial = device['c8y_Hardware']['serialNumber']
            except (KeyError, TypeError):
                serial = 'N/A'
            device_id = device.id
            display_name = f"{name} (S/N: {serial})"
            devices_structured_list.append({
                'display': display_name,
                'name': name,
                'serial': serial,
                'id': device_id
            })
        return sorted(devices_structured_list, key=lambda d: d['display'])
    except Exception as e:
        st.error(f"Erro ao buscar dispositivos: {e}")
        return []


@st.cache_data(ttl=300)
def fetch_supported_series(tenant, user, password, device_id):
    try:
        full_tenant_id = tenant.split('.')[0].split('//')[1]
        if '-' in full_tenant_id:
            formatted_user = f"{full_tenant_id}/{user}"
        else:
            formatted_user = user
        c8y = CumulocityApi(base_url=tenant, tenant_id=full_tenant_id, username=formatted_user, password=password)
        endpoint = f'/inventory/managedObjects/{device_id}/supportedSeries'
        response_json = c8y.get(endpoint)
        return response_json.get('c8y_SupportedSeries', [])
    except Exception as e:
        st.error(f"Erro ao buscar medições suportadas: {e}")
        return []


# --- Lógica de Análise (Backend) ---

def _find_operational_threshold(points, log_queue, device_display_name, measurement_name):
    """
    Usa clustering (K-Means) para separar os dados em dois grupos (operacional e não-operacional)
    e retorna o limiar que os divide.
    """
    if not points or len(points) < 10:
        return None

    try:
        log_queue.put({'type': 'log',
                       'data': f"[{device_display_name}] Calculando limiar automático para {measurement_name}..."})

        values = np.array([p[1] for p in points]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(values)

        low_cluster_label = np.argmin(kmeans.cluster_centers_)
        low_cluster_values = values[kmeans.labels_ == low_cluster_label]

        if low_cluster_values.size > 0:
            threshold = low_cluster_values.max()
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name}] Limiar automático para {measurement_name} definido em: {threshold:.4f}",
                           'color': 'success'})
            return threshold
        return None
    except Exception as e:
        log_queue.put({'type': 'log',
                       'data': f"AVISO: Falha ao calcular limiar automático para {measurement_name}. Causa: {e}",
                       'color': 'warning'})
        return None


def _fetch_all_raw_data(c8y, device_id, measurements_to_fetch, date_from, date_to, log_queue, device_display_name):
    """Busca todas as medições brutas necessárias para a análise de um dispositivo."""
    raw_data = {}
    api_call_counter = 0
    for measurement_name in measurements_to_fetch:
        log_queue.put({'type': 'log',
                       'data': f"[{device_display_name}] Buscando dados para: {measurement_name}..."})
        measurements = list(
            c8y.measurements.select(source=device_id, type=measurement_name, date_from=date_from,
                                    date_to=date_to))
        api_call_counter += 1
        points = [(datetime.fromisoformat(m.time.replace("Z", "+00:00")),
                   extract_measurement_value(m, measurement_name)) for m in measurements if
                  extract_measurement_value(m, measurement_name) is not None]
        raw_data[measurement_name] = sorted(points, key=lambda x: x[0])
    return raw_data, api_call_counter


def _processar_ciclos_operacionais(raw_data, device_config: DeviceAnalysisConfig, date_from_str: str, date_to_str: str,
                                   log_queue, job_label):
    """Identifica ciclos operacionais com base nas medições de gatilho e calcula KPIs básicos."""
    log_queue.put({'type': 'log',
                   'data': f"[{device_config.device_display_name} | {job_label}] Processando ciclos com base em: {device_config.load_measurement_names}"})

    all_points = []
    for trigger_name in device_config.load_measurement_names:
        if trigger_name in raw_data:
            for ts, val in raw_data[trigger_name]:
                all_points.append({'time': ts, 'value': val, 'type': trigger_name})
    all_points.sort(key=lambda p: p['time'])

    if not all_points:
        return [], {}, 0

    summed_trigger_measurements = []
    last_known_values = {name: 0.0 for name in device_config.load_measurement_names}
    for point in all_points:
        last_known_values[point['type']] = point['value']
        current_sum = sum(last_known_values.values())
        summed_trigger_measurements.append((point['time'], current_sum))

    operational_cycles = []
    operational_kpis = {}
    if summed_trigger_measurements:
        cycle_start_time = None
        for ts, summed_value in summed_trigger_measurements:
            is_on = summed_value > device_config.operating_current
            if is_on and cycle_start_time is None:
                cycle_start_time = ts
            elif not is_on and cycle_start_time is not None:
                operational_cycles.append({"start": cycle_start_time, "end": ts})
                cycle_start_time = None
        if cycle_start_time is not None:
            last_ts = summed_trigger_measurements[-1][0]
            operational_cycles.append({"start": cycle_start_time, "end": last_ts})

    log_queue.put({'type': 'log',
                   'data': f"[{device_config.device_display_name} | {job_label}] Mapeamento concluído. {len(operational_cycles)} ciclos encontrados."})

    operational_kpis['num_cycles'] = len(operational_cycles)
    total_uptime_seconds = sum((c['end'] - c['start']).total_seconds() for c in operational_cycles)
    operational_kpis['total_uptime'] = total_uptime_seconds

    if operational_kpis['num_cycles'] > 0:
        operational_kpis['mean_cycle_duration'] = total_uptime_seconds / operational_kpis['num_cycles']

    if operational_kpis['num_cycles'] > 1:
        total_off_time_seconds = sum(
            (operational_cycles[i + 1]['start'] - operational_cycles[i]['end']).total_seconds() for i in
            range(len(operational_cycles) - 1))
        operational_kpis['mean_time_between_cycles'] = total_off_time_seconds / (
                operational_kpis['num_cycles'] - 1)

    date_from_obj = datetime.strptime(date_from_str, '%Y-%m-%d')
    date_to_obj = datetime.strptime(date_to_str, '%Y-%m-%d') + timedelta(days=1)
    total_analysis_duration_seconds = (date_to_obj - date_from_obj).total_seconds()

    if total_analysis_duration_seconds > 0:
        operational_kpis['duty_cycle'] = (total_uptime_seconds / total_analysis_duration_seconds) * 100

    return operational_cycles, operational_kpis, total_analysis_duration_seconds


def _calcular_kpis_de_confiabilidade(operational_cycles, alarms, total_analysis_duration_seconds, log_queue,
                                     device_display_name, job_label):
    """Calcula KPIs de disponibilidade e falhas com base nos ciclos e alarmes."""
    kpis = {}
    if not operational_cycles or not alarms:
        kpis['availability'] = 100.0
        kpis['downtime_due_to_fault'] = 0
        kpis['number_of_faults'] = 0
        return kpis

    log_queue.put(
        {'type': 'log', 'data': f"[{device_display_name} | {job_label}] Calculando disponibilidade..."})
    alarm_timestamps = sorted([pd.to_datetime(a['time']) for a in alarms])

    total_downtime_due_to_fault_seconds = 0
    number_of_faults = 0

    for i in range(len(operational_cycles) - 1):
        cycle_end_time = operational_cycles[i]['end']
        next_cycle_start_time = operational_cycles[i + 1]['start']

        fault_window_start = cycle_end_time - timedelta(minutes=1)
        is_fault_stop = any(
            fault_window_start <= alarm_time <= cycle_end_time for alarm_time in alarm_timestamps)

        if is_fault_stop:
            downtime_duration = (next_cycle_start_time - cycle_end_time).total_seconds()
            total_downtime_due_to_fault_seconds += downtime_duration
            number_of_faults += 1

    if total_analysis_duration_seconds > 0:
        availability = ((
                                total_analysis_duration_seconds - total_downtime_due_to_fault_seconds) / total_analysis_duration_seconds) * 100
        kpis['availability'] = availability

    kpis['downtime_due_to_fault'] = total_downtime_due_to_fault_seconds
    kpis['number_of_faults'] = number_of_faults
    return kpis


def _analisar_dados_nos_ciclos(raw_data, operational_cycles, device_config: DeviceAnalysisConfig):
    """Analisa as medições alvo dentro dos períodos de estabilização de cada ciclo."""
    results_data = {
        target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None, "all_values": []}
        for target in device_config.target_measurements_list}

    for cycle in operational_cycles:
        analysis_start = cycle['start'] + timedelta(seconds=device_config.stabilization_delay)
        analysis_end = cycle['end'] - timedelta(seconds=device_config.shutdown_delay)
        if analysis_start >= analysis_end: continue

        for target_name in device_config.target_measurements_list:
            for time_obj, value in raw_data.get(target_name, []):
                if analysis_start <= time_obj <= analysis_end:
                    res = results_data[target_name]
                    if res["min"] is None or value < res["min"]: res["min"], res["min_time"] = value, time_obj
                    if res["max"] is None or value > res["max"]: res["max"], res["max_time"] = value, time_obj
                    res["count_valid"] += 1
                    res["all_values"].append(value)
    return results_data


def _analisar_assinatura_de_partida(raw_data, operational_cycles, device_config: DeviceAnalysisConfig, log_queue,
                                    job_label):
    """Processa e analisa as curvas de partida do motor."""
    startup_analysis = {}
    motor_measurements = [m for m in device_config.load_measurement_names if m.startswith('MA_')]

    for motor_measurement in motor_measurements:
        if motor_measurement in raw_data and raw_data[motor_measurement]:
            log_queue.put({'type': 'log',
                           'data': f"[{device_config.device_display_name} | {job_label}] Analisando partidas de {motor_measurement}..."})
            startup_curves = []

            # --- CORREÇÃO DO ERRO DE DUPLICATAS ---
            # Se houver timestamps duplicados nos dados brutos, isso causa um erro de reindexação mais tarde.
            # A correção agrupa por timestamp e calcula a média para garantir um índice único.
            df_ma_temp = pd.DataFrame(raw_data[motor_measurement], columns=['time', motor_measurement])
            df_ma = df_ma_temp.groupby('time').mean()

            for cycle in operational_cycles:
                startup_window_end = cycle['start'] + timedelta(seconds=device_config.startup_duration)
                curve_df = df_ma[(df_ma.index >= cycle['start']) & (df_ma.index <= startup_window_end)]

                if not curve_df.empty:
                    curve_df = curve_df.copy()
                    curve_df['relative_time'] = (curve_df.index - cycle['start']).total_seconds()
                    startup_curves.append(curve_df.set_index('relative_time')[motor_measurement])

            if startup_curves:
                try:
                    resample_index = pd.to_timedelta(np.arange(0, device_config.startup_duration, 0.1),
                                                     unit='s')
                    resampled_curves = [
                        s.reindex(pd.to_timedelta(s.index, unit='s').union(resample_index)).interpolate(
                            method='time').reindex(resample_index) for s in startup_curves]

                    combined_df = pd.concat(resampled_curves, axis=1)
                    if not combined_df.empty:
                        combined_df.index = combined_df.index.total_seconds()
                        startup_analysis[motor_measurement] = {
                            'mean': combined_df.mean(axis=1).to_dict(),
                            'std': combined_df.std(axis=1).to_dict(),
                            'curves': [s.dropna().to_dict() for s in startup_curves]
                        }
                except Exception as e:
                    log_queue.put({'type': 'log',
                                   'data': f"AVISO: Falha ao processar curvas de partida para {motor_measurement}. Causa: {e}",
                                   'color': 'warning'})
    return startup_analysis


def _calcular_relacao_compressao(raw_data, results_data, operational_kpis, log_queue, device_display_name, job_label):
    """
    Busca dinamicamente por pares de medições de pressão de sucção (SP_xx) e descarga (DP_xx)
    e calcula a Relação de Compressão para cada par encontrado.
    """
    sufixos = set()
    measurement_names = list(raw_data.keys())
    for name in measurement_names:
        match = re.search(r'_(?P<sufixo>\d+)$', name)
        if match:
            sufixos.add(match.group('sufixo'))

    for sufixo in sufixos:
        sp_name = f"SP_{sufixo}"
        dp_name = f"DP_{sufixo}"

        if sp_name in raw_data and dp_name in raw_data and raw_data[sp_name] and raw_data[dp_name]:
            ratio_name = f"Relação de Compressão {sufixo}"
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name} | {job_label}] Calculando {ratio_name}..."})
            try:
                df_dp = pd.DataFrame(raw_data[dp_name], columns=['time', dp_name]).set_index('time')
                df_sp = pd.DataFrame(raw_data[sp_name], columns=['time', sp_name]).set_index('time')
                df_aligned = pd.concat([df_dp, df_sp], axis=1).interpolate(method='time').dropna()

                if not df_aligned.empty:
                    df_aligned['ratio'] = (df_aligned[dp_name] + 1.013) / (df_aligned[sp_name] + 1.013)
                    ratio_series = df_aligned['ratio']

                    if not ratio_series.empty:
                        results_data[ratio_name] = {
                            "min": ratio_series.min(),
                            "max": ratio_series.max(),
                            "count_valid": len(ratio_series),
                            "min_time": ratio_series.idxmin(),
                            "max_time": ratio_series.idxmax(),
                            "all_values": ratio_series.tolist()
                        }
                        operational_kpis.setdefault('mean_values', {})[ratio_name] = ratio_series.mean()
                        raw_data[ratio_name] = list(ratio_series.reset_index().to_records(index=False))

            except Exception as e:
                log_queue.put({'type': 'log',
                               'data': f"AVISO: Não foi possível calcular {ratio_name}. Causa: {e}",
                               'color': 'warning'})

    return results_data, operational_kpis, raw_data


def _analisar_alarmes_recorrentes(alarms_and_events, log_queue, device_display_name, job_label):
    """Analisa a frequência e o MTBA dos alarmes."""
    alarm_analysis = {}
    if not alarms_and_events['alarms']:
        return alarm_analysis

    log_queue.put(
        {'type': 'log', 'data': f"[{device_display_name} | {job_label}] Analisando alarmes recorrentes..."})
    df_alarms = pd.DataFrame(alarms_and_events['alarms'])
    df_alarms['time'] = pd.to_datetime(df_alarms['time'])
    df_alarms['base_text'] = df_alarms['text'].apply(lambda x: x.split(' - "')[0] if ' - "' in x else x)
    alarm_counts = df_alarms['base_text'].value_counts().reset_index()
    alarm_counts.columns = ['Alarme', 'Ocorrências']
    df_alarms = df_alarms.sort_values('time')
    mtba_list = []
    for alarm_text, group in df_alarms.groupby('base_text'):
        if len(group) > 1:
            diffs = group['time'].diff().dt.total_seconds()
            mtba_seconds = diffs.mean()
            mtba_list.append({'Alarme': alarm_text, 'MTBA_seconds': mtba_seconds})

    if mtba_list:
        df_mtba = pd.DataFrame(mtba_list)
        final_alarm_df = pd.merge(alarm_counts, df_mtba, on='Alarme', how='left')
    else:
        final_alarm_df = alarm_counts
        final_alarm_df['MTBA_seconds'] = pd.NA

    final_alarm_df['MTBA'] = final_alarm_df['MTBA_seconds'].apply(format_uptime)
    alarm_analysis['ranking'] = final_alarm_df.to_dict('records')
    type_counts = df_alarms['type'].value_counts().reset_index()
    type_counts.columns = ['Tipo', 'Ocorrências']
    alarm_analysis['by_type'] = type_counts.to_dict('records')
    return alarm_analysis


def _calculate_trend_indicators(points, period_duration_days):
    """
    Calcula indicadores de tendência para uma série de dados de medição (MKPRED).
    """
    if not points or len(points) < 2:
        return {
            'std_dev': 0, 'slope': 0, 'intercept': 0,
            'r_squared': 0, 'rate_of_change_day': 0, 'mean': 0,
            'growth_in_period': 0, 'first_value': 0, 'last_value': 0,
            'growth_rate_period': 0
        }

    timestamps, values = zip(*points)
    series = pd.Series(values)

    numeric_time = [(t - timestamps[0]).total_seconds() for t in timestamps]

    std_dev = series.std()
    mean_value = series.mean()

    slope, intercept, r_value, p_value, std_err = stats.linregress(numeric_time, values)
    r_squared = r_value ** 2

    change_per_day = slope * 86400

    if abs(mean_value) > 1e-6:
        rate_of_change_day = (change_per_day / mean_value) * 100
    else:
        rate_of_change_day = float('inf') if change_per_day > 0 else 0

    # --- CÁLCULO DA EVOLUÇÃO PERCENTUAL NO PERÍODO ---
    first_value = values[0]
    last_value = values[-1]
    growth_in_period = 0
    if abs(first_value) > 1e-9:
        growth_in_period = ((last_value - first_value) / abs(first_value)) * 100

    # --- NOVO CÁLCULO: CRESCIMENTO MÉDIO NO PERÍODO (BASEADO NA TENDÊNCIA) ---
    growth_rate_period = rate_of_change_day * period_duration_days

    return {
        'std_dev': std_dev,
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'rate_of_change_day': rate_of_change_day,
        'mean': mean_value,
        'growth_in_period': growth_in_period,
        'first_value': first_value,
        'last_value': last_value,
        'growth_rate_period': growth_rate_period
    }


def _calculate_predictive_health_index(indicators, measurement_name, current_value, config: DeviceAnalysisConfig):
    """
    Calcula o Índice de Saúde Preditivo para um dispositivo MKPRED com limites e pesos configuráveis.
    """
    if not indicators or current_value is None:
        return 0

    # Pega o limite da configuração, se não existir, retorna um valor neutro.
    limit = config.measurement_limits.get(measurement_name)
    if limit is None or limit == 0:
        return 50  # Retorna um valor neutro se nenhum limite for configurado

    # --- Componente de Severidade ---
    percentage_of_limit = (current_value / limit) * 100
    severity_score = 100 - percentage_of_limit
    severity_score = max(0, min(100, severity_score))

    # --- Componente de Degradação ---
    rate_of_change_day = indicators.get('rate_of_change_day', 0)

    # Limites para a taxa de degradação (pode ser configurável no futuro)
    stable_roc_threshold = 0.5
    critical_roc_threshold = 5.0

    if rate_of_change_day <= stable_roc_threshold:
        base_degradation_score = 100
    elif rate_of_change_day > critical_roc_threshold:
        base_degradation_score = 0
    else:
        # Interpolação linear entre os limiares
        base_degradation_score = 100 - (((rate_of_change_day - stable_roc_threshold) / (
                critical_roc_threshold - stable_roc_threshold)) * 100)

    r_squared = indicators.get('r_squared', 0)
    # A penalidade é a pontuação perdida, multiplicada pela confiança na tendência
    penalty = (100 - base_degradation_score) * r_squared
    degradation_score = 100 - penalty
    degradation_score = max(0, min(100, degradation_score))

    # --- Cálculo Final Ponderado ---
    weights = config.health_kpi_weights
    final_score = (severity_score * weights['severity']) + (degradation_score * weights['degradation'])

    return max(0, min(100, final_score))


def get_trend_status(health_index):
    """Retorna uma string formatada para o status da tendência com base no índice de saúde."""
    if health_index >= 80:
        return "🟢 Estável"
    elif 60 <= health_index < 80:
        return "🟡 Atenção"
    else:
        return "🔴 Alerta Crítico"


def _analisar_assinatura_de_ciclo_completo(raw_data, operational_cycles, device_config: DeviceAnalysisConfig, log_queue,
                                           job_label):
    """
    Processa e analisa a assinatura do ciclo de operação completo para medições de carga.
    """
    cycle_signature_analysis = {}
    motor_measurement = next((m for m in device_config.load_measurement_names if m.startswith('MA_')), None)

    if not motor_measurement or not operational_cycles:
        return cycle_signature_analysis

    if motor_measurement in raw_data and raw_data[motor_measurement]:
        log_queue.put({'type': 'log',
                       'data': f"[{device_config.device_display_name} | {job_label}] Analisando Assinatura de Ciclo Completo para {motor_measurement}..."})

        all_cycle_curves = []
        df_motor = pd.DataFrame(raw_data[motor_measurement], columns=['time', motor_measurement]).set_index('time')

        for i, cycle in enumerate(operational_cycles):
            cycle_df = df_motor[(df_motor.index >= cycle['start']) & (df_motor.index <= cycle['end'])]

            if not cycle_df.empty:
                cycle_df = cycle_df.copy()
                cycle_duration = (cycle['end'] - cycle['start']).total_seconds()
                if cycle_duration == 0: continue

                cycle_df['normalized_time'] = ((cycle_df.index - cycle['start']).total_seconds() / cycle_duration) * 100
                original_curve = cycle_df.set_index('normalized_time')[motor_measurement]
                all_cycle_curves.append({'id': i, 'curve': original_curve})

        if all_cycle_curves:
            try:
                resample_index = np.linspace(0, 100, 101)
                resampled_curves_dict = {}
                for cycle_data in all_cycle_curves:
                    s = cycle_data['curve']
                    resampled_s = np.interp(resample_index, s.index, s.values)
                    resampled_curves_dict[f"ciclo_{cycle_data['id']}"] = resampled_s

                combined_df = pd.DataFrame(resampled_curves_dict, index=resample_index)

                if not combined_df.empty:
                    mean_curve = combined_df.mean(axis=1)
                    std_curve = combined_df.std(axis=1)

                    cycle_signature_analysis[motor_measurement] = {
                        'mean': mean_curve.to_dict(),
                        'std': std_curve.to_dict(),
                        'upper_bound': (mean_curve + std_curve).to_dict(),
                        'lower_bound': (mean_curve - std_curve).to_dict(),
                        'curves': {f"ciclo_{cd['id']}": cd['curve'].to_dict() for cd in all_cycle_curves}
                    }
            except Exception as e:
                log_queue.put({'type': 'log',
                               'data': f"AVISO: Falha ao processar assinatura de ciclo completo para {motor_measurement}. Causa: {e}",
                               'color': 'warning'})

    return cycle_signature_analysis


def _sugerir_correlacoes(raw_data, log_queue, device_display_name):
    """Analisa e sugere as correlações mais fortes entre as medições."""
    correlation_suggestions = []
    valid_series = {name: data for name, data in raw_data.items() if len(data) > 10}
    if len(valid_series) < 2:
        return correlation_suggestions

    log_queue.put({'type': 'log', 'data': f"[{device_display_name}] Calculando correlações inteligentes..."})

    df_list = []
    # --- CORREÇÃO DO ERRO DE ÍNDICE DUPLICADO ---
    # Garante que cada série tenha um índice de tempo único antes de concatenar
    for name, data in valid_series.items():
        temp_df = pd.DataFrame(data, columns=['time', name])
        # Agrupa por timestamp e calcula a média para remover duplicatas
        unique_df = temp_df.groupby('time').mean()
        df_list.append(unique_df)

    if not df_list:
        return correlation_suggestions

    aligned_df = pd.concat(df_list, axis=1).interpolate(method='time').dropna()

    if len(aligned_df) < 2:
        return correlation_suggestions

    corr_matrix = aligned_df.corr().abs()
    sol = corr_matrix.unstack()
    so = sol.sort_values(kind="quicksort", ascending=False)

    seen_pairs = set()
    for (idx, val) in so.items():
        if idx[0] == idx[1]:
            continue

        pair = tuple(sorted((idx[0], idx[1])))
        if pair not in seen_pairs:
            if val > 0.7:
                correlation_suggestions.append({'pair': f"{pair[0]} & {pair[1]}", 'value': val})
            seen_pairs.add(pair)

        if len(correlation_suggestions) >= 3:
            break

    return correlation_suggestions


# --- FUNÇÃO PRINCIPAL REATORADA ---
def analyze_single_device(job: AnalysisJob, log_queue: Queue):
    """Função principal que orquestra a análise de um único dispositivo."""
    job_label = job.job_label
    device_config = job.device_config
    device_id = device_config.device_id
    device_display_name = device_config.device_display_name
    api_call_counter = 0

    try:
        full_tenant_id = job.connection.tenant_url.split('.')[0].split('//')[1]
        # Lógica de autenticação aprimorada para subtenants
        if '-' in full_tenant_id:  # Heurística para subtenant
            formatted_user = f"{full_tenant_id}/{job.connection.username}"
        else:  # Tenant principal
            formatted_user = job.connection.username
        c8y = CumulocityApi(base_url=job.connection.tenant_url,
                            tenant_id=full_tenant_id,
                            username=formatted_user, password=job.connection.password)

        log_queue.put({'type': 'log', 'data': f"[{device_display_name} | {job_label}] Iniciando análise..."})

        all_measurements_to_fetch = set(device_config.target_measurements_list)
        if not device_config.is_mkpred:
            all_measurements_to_fetch.update(device_config.load_measurement_names)

        raw_data, api_calls = _fetch_all_raw_data(c8y, device_id, all_measurements_to_fetch, job.date_from,
                                                  job.date_to, log_queue, device_display_name)
        api_call_counter += api_calls

        alarms_and_events = {'alarms': [], 'events': []}
        if job.fetch_alarms:
            alarms = c8y.alarms.select(source=device_id, date_from=job.date_from, date_to=job.date_to)
            api_call_counter += 1
            for a in alarms:
                alarms_and_events['alarms'].append(
                    {'time': a.time, 'text': a.text, 'type': a.type, 'severity': a.severity})

        results_data = {}
        operational_kpis = {}
        startup_analysis = {}
        trend_analysis = {}
        cycle_signature_analysis = {}
        correlation_suggestions = _sugerir_correlacoes(raw_data, log_queue, device_display_name)

        if device_config.is_mkpred:
            log_queue.put({'type': 'log',
                           'data': f"[{device_display_name} | {job_label}] Modo MKPRED: analisando período completo."})

            date_from_obj = datetime.strptime(job.date_from, '%Y-%m-%d')
            date_to_obj = datetime.strptime(job.date_to, '%Y-%m-%d')
            period_duration_days = (date_to_obj - date_from_obj).days + 1

            results_data = {
                target: {"min": None, "max": None, "count_valid": 0, "min_time": None, "max_time": None,
                         "all_values": []}
                for target in device_config.target_measurements_list}

            for target_name in device_config.target_measurements_list:
                points = raw_data.get(target_name, [])

                if device_config.operation_filter_mode != 'none':
                    threshold = None
                    if device_config.operation_filter_mode == 'auto':
                        threshold = _find_operational_threshold(points, log_queue, device_display_name, target_name)
                    elif device_config.operation_filter_mode == 'manual':
                        threshold = device_config.manual_thresholds.get(target_name)
                        if threshold is not None:
                            log_queue.put({'type': 'log',
                                           'data': f"[{device_display_name}] Usando limiar manual de {threshold} para {target_name}"})

                    if threshold is not None:
                        original_count = len(points)
                        if original_count > 0:
                            points = [p for p in points if p[1] > threshold]
                            log_queue.put({'type': 'log',
                                           'data': f"[{device_display_name}] Filtro aplicado em {target_name}: {original_count} -> {len(points)} pontos."})

                if points:
                    timestamps, values = zip(*points)
                    series = pd.Series(values)
                    results_data[target_name].update({
                        "min": series.min(), "max": series.max(), "count_valid": len(series),
                        "min_time": timestamps[series.idxmin()], "max_time": timestamps[series.idxmax()],
                        "all_values": values
                    })
                    trend_indicators = _calculate_trend_indicators(points, period_duration_days)

                    current_value = series.mean()
                    health_index = _calculate_predictive_health_index(trend_indicators, target_name, current_value,
                                                                      device_config)
                    trend_indicators['health_index'] = health_index
                    trend_indicators['status'] = get_trend_status(health_index)
                    trend_analysis[target_name] = trend_indicators

            operational_kpis = {'is_mkpred': True}

        else:  # Lógica para compressores
            operational_cycles, operational_kpis, total_duration = _processar_ciclos_operacionais(raw_data,
                                                                                                  device_config,
                                                                                                  job.date_from,
                                                                                                  job.date_to,
                                                                                                  log_queue, job_label)

            if not operational_cycles:
                log_queue.put({'type': 'log',
                               'data': f"AVISO: [{device_display_name} | {job_label}] Nenhum ciclo operacional encontrado.",
                               'color': 'warning'})
            else:
                results_data = _analisar_dados_nos_ciclos(raw_data, operational_cycles, device_config)

                if device_config.refrigeration_limit_mode == 'auto':
                    log_queue.put({'type': 'log',
                                   'data': f"[{device_display_name}] Calculando limites de performance automaticamente..."})
                    for tm, data in results_data.items():
                        if 'all_values' in data and data['all_values']:
                            series = pd.Series(data['all_values'])
                            p10 = series.quantile(0.10)
                            p90 = series.quantile(0.90)
                            if p90 > p10:
                                device_config.refrigeration_kpi_limits[tm] = {'min': p10, 'max': p90}
                                log_queue.put({'type': 'log',
                                               'data': f"[{device_display_name}] Limites para {tm} definidos: Mín({p10:.2f}), Máx({p90:.2f})"})

                startup_analysis = _analisar_assinatura_de_partida(raw_data, operational_cycles, device_config,
                                                                   log_queue, job_label)
                cycle_signature_analysis = _analisar_assinatura_de_ciclo_completo(raw_data, operational_cycles,
                                                                                  device_config, log_queue, job_label)

                kpis_confiabilidade = _calcular_kpis_de_confiabilidade(operational_cycles, alarms_and_events['alarms'],
                                                                       total_duration, log_queue, device_display_name,
                                                                       job_label)
                operational_kpis.update(kpis_confiabilidade)
                results_data, operational_kpis, raw_data = _calcular_relacao_compressao(raw_data, results_data,
                                                                                        operational_kpis, log_queue,
                                                                                        device_display_name, job_label)

        alarm_analysis = _analisar_alarmes_recorrentes(alarms_and_events, log_queue, device_display_name, job_label)

        operational_kpis.setdefault('mean_values', {})
        operational_kpis.setdefault('std_dev_values', {})
        for target_name, data in results_data.items():
            if data.get('all_values'):
                series = pd.Series(data['all_values'])
                if not series.empty:
                    data['mean'] = series.mean()
                    data['median'] = series.median()
                    data['std_dev'] = series.std()
                    data['range'] = data['max'] - data['min'] if data['max'] is not None and data[
                        'min'] is not None else 0
                    data['p95'] = series.quantile(0.95)

                    operational_kpis['mean_values'][target_name] = data['mean']
                    operational_kpis['std_dev_values'][target_name] = data['std_dev']
                else:
                    data.update({'mean': 0, 'median': 0, 'std_dev': 0, 'range': 0, 'p95': 0})
                    operational_kpis['mean_values'][target_name] = 0
                    operational_kpis['std_dev_values'][target_name] = 0
            if 'all_values' in data:
                del data['all_values']

        if not device_config.is_mkpred:
            operational_kpis['health_index'] = calculate_health_index(operational_kpis, device_config)

        return job_label, device_display_name, results_data, raw_data, api_call_counter, operational_kpis, alarms_and_events, alarm_analysis, startup_analysis, trend_analysis, cycle_signature_analysis, correlation_suggestions

    except Exception as e:
        import traceback
        log_queue.put(
            {'type': 'log',
             'data': f"ERRO FATAL ao analisar {device_display_name} ({job_label}): {e}\n{traceback.format_exc()}",
             'color': 'error'})
        return job_label, device_display_name, {}, {}, api_call_counter, {}, {}, {}, {}, {}, {}, []


def perform_analysis_master_thread(stop_event, log_queue, jobs_to_run: List[AnalysisJob]):
    total_api_calls = 0
    final_results, final_raw_data, final_kpis, final_alarms_events = {}, {}, {}, {}
    final_alarm_analysis, final_startup_analysis, final_trend_analysis = {}, {}, {}
    final_cycle_signature_analysis, final_correlation_suggestions = {}, {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_job = {executor.submit(analyze_single_device, job, log_queue): job for job in jobs_to_run}

        for i, future in enumerate(as_completed(future_to_job)):
            if stop_event.is_set():
                log_queue.put({'type': 'log', 'data': "Cancelamento solicitado.", 'color': 'warning'})
                break

            job_label, device_name, results, raw, api_calls, kpis, alarms_events, alarm_analysis, startup_analysis, trend_analysis, cycle_signature, corr_sugg = future.result()

            final_results.setdefault(job_label, {})[device_name] = results
            final_raw_data.setdefault(job_label, {})[device_name] = raw
            final_kpis.setdefault(job_label, {})[device_name] = kpis
            final_alarms_events.setdefault(job_label, {})[device_name] = alarms_events
            final_alarm_analysis.setdefault(job_label, {})[device_name] = alarm_analysis
            final_startup_analysis.setdefault(job_label, {})[device_name] = startup_analysis
            final_trend_analysis.setdefault(job_label, {})[device_name] = trend_analysis
            final_cycle_signature_analysis.setdefault(job_label, {})[device_name] = cycle_signature
            final_correlation_suggestions.setdefault(job_label, {})[device_name] = corr_sugg

            total_api_calls += api_calls
            log_queue.put(
                {'type': 'status', 'data': f"Análise concluída para {i + 1}/{len(jobs_to_run)} jobs.",
                 'progress': (i + 1) / len(jobs_to_run)})

    log_queue.put({'type': 'log', 'data': f"Análise concluída. Total de Chamadas à API: {total_api_calls}."})
    log_queue.put({'type': 'finished',
                   'data': {'results': final_results, 'raw': final_raw_data, 'api_calls': total_api_calls,
                            'kpis': final_kpis, 'alarms_events': final_alarms_events,
                            'alarm_analysis': final_alarm_analysis,
                            'startup_analysis': final_startup_analysis,
                            'trend_analysis': final_trend_analysis,
                            'cycle_signature_analysis': final_cycle_signature_analysis,
                            'correlation_suggestions': final_correlation_suggestions
                            }})


# --- Funções de UI (Refatoradas) ---
def run_tour():
    """Executa uma sequência de toasts para guiar o utilizador."""
    st.toast("Bem-vindo! Este é o painel de configurações. ⚙️", icon="👋")
    time.sleep(3)
    st.toast("Aqui você conecta, seleciona dispositivos e define os parâmetros da análise.", icon="🔩")
    time.sleep(4)
    st.toast("Após a análise, os resultados são exibidos aqui, na área principal. 📈", icon="➡️")
    time.sleep(4)
    st.toast("Use as abas para navegar entre os dispositivos e personalizar a sua visualização. Bom trabalho!",
             icon="👍")


def display_configuration_sidebar():
    """Renderiza toda a barra lateral de configuração."""
    with st.sidebar:
        st.header("⚙️ Configurações da Análise")

        if st.button("❔ Iniciar Tour Guiado", use_container_width=True):
            run_tour()

        with st.expander("1. Conexão com a Plataforma", expanded=True):
            tenant = st.text_input("Tenant (URL)",
                                   value=st.session_state.get('tenant', "https://mayekawa.us.cumulocity.com"),
                                   key='tenant')
            username = st.text_input("Username", value=st.session_state.get('username', ""), key='username')
            password = st.text_input("Password", type="password")

            if st.button("Conectar e Listar Dispositivos"):
                if username and password:
                    with st.spinner("Buscando dispositivos..."):
                        st.session_state.structured_device_list = fetch_devices(tenant, username, password)
                else:
                    st.warning("Por favor, preencha Username e Password.")

        if 'structured_device_list' not in st.session_state or not st.session_state.structured_device_list:
            st.info("Conecte-se a uma plataforma para carregar os dispositivos.")
            return

        with st.expander("2. Modo de Análise e Seleção", expanded=True):
            analysis_mode = st.radio(
                "Escolha o que deseja fazer:",
                ["Análise Detalhada", "Comparar Dispositivos"],
                key='analysis_mode',
                horizontal=True
            )

            filter_name_serial = st.text_input("Filtrar por Nome ou S/N", key="filter_name")
            filtered_list = [d for d in st.session_state.structured_device_list if filter_name_serial.lower() in d[
                'display'].lower()] if filter_name_serial else st.session_state.structured_device_list
            display_options = [d['display'] for d in filtered_list]

            selected_devices_display = st.multiselect("Selecione os Dispositivos", display_options,
                                                      key='selected_devices_display')

        if not selected_devices_display:
            st.warning("Selecione pelo menos um dispositivo para continuar.")
            return

        with st.expander("3. Períodos e Parâmetros de Análise", expanded=True):
            all_device_configs = {}

            col1, col2 = st.columns(2)
            with col1:
                date_from = st.date_input("Data de Início", datetime.now() - timedelta(days=7))
            with col2:
                date_to = st.date_input("Data de Fim", datetime.now())

            st.markdown("---")
            device_tabs = st.tabs(selected_devices_display)
            for i, device_tab in enumerate(device_tabs):
                with device_tab:
                    current_device_display = selected_devices_display[i]
                    current_device_obj = next((d for d in filtered_list if d['display'] == current_device_display),
                                              None)
                    if not current_device_obj: continue

                    device_id = current_device_obj['id']
                    series_list = fetch_supported_series(tenant, username, password, device_id)
                    cleaned_series_names = sorted(list(set([s.split('.')[0] for s in series_list])))
                    is_mkpred = is_likely_mkpred(series_list)

                    default_targets = [n for n in
                                       ['SP_01', 'DP_01', 'OT_01', 'DT_01', 'MA_01', 'v_rms', 'a_rms', 'a_peak'] if
                                       n in cleaned_series_names]
                    target_measurements = st.multiselect("Medições Alvo", options=cleaned_series_names,
                                                         default=default_targets, key=f"targets_{device_id}")

                    measurement_limits = {}
                    health_kpi_weights = {'severity': 0.4, 'degradation': 0.6}
                    manual_thresholds = {}
                    refrigeration_kpi_limits = {}
                    refrigeration_kpi_weights = {'availability': 0.5, 'stability': 0.3, 'performance': 0.2}
                    acceptable_variation_percent = 10.0
                    refrigeration_limit_mode = 'manual'

                    if is_mkpred:
                        st.info("Dispositivo de vibração (MKPRED) detectado. A análise será de tendência contínua.")

                        with st.expander("Filtro de Operação (Vibração)"):
                            filter_mode_map = {
                                "Usar limiar automático (Recomendado)": "auto",
                                "Definir limiar manual": "manual",
                                "Nenhum filtro": "none"
                            }
                            filter_mode_display = st.radio("Como tratar dados de baixa atividade?",
                                                           options=filter_mode_map.keys(),
                                                           key=f"filter_mode_{device_id}")
                            operation_filter_mode = filter_mode_map[filter_mode_display]

                            if operation_filter_mode == 'manual':
                                st.markdown("**Limiares Manuais de Operação**")
                                for tm in target_measurements:
                                    default_thresh = 0.1 if 'VEL' in tm else 100.0 if 'AC' in tm else 0.0
                                    thresh = st.number_input(f"Limiar para {tm}", min_value=0.0, value=default_thresh,
                                                             step=0.01, format="%.4f",
                                                             key=f"manual_thresh_{device_id}_{tm}")
                                    manual_thresholds[tm] = thresh

                        with st.expander("⚙️ Configurações de KPI Preditivo"):
                            st.markdown("**Limites de Alerta por Medição**")
                            for tm in target_measurements:
                                default_val = 4.0 if 'VEL' in tm else 2500.0 if 'AC' in tm else 0.0
                                limit = st.number_input(f"Limite para {tm}", min_value=0.0, value=default_val,
                                                        key=f"limit_{device_id}_{tm}", format="%.2f")
                                measurement_limits[tm] = limit

                            st.markdown("**Pesos do Índice de Saúde**")
                            w_sev = st.slider("Peso da Severidade", 0, 100, 40, 5, key=f"w_sev_{device_id}")
                            health_kpi_weights['severity'] = w_sev / 100.0
                            health_kpi_weights['degradation'] = 1.0 - (w_sev / 100.0)
                            st.write(f"Peso da Degradação: **{health_kpi_weights['degradation']:.0%}**")

                        device_config = DeviceAnalysisConfig(
                            device_id=device_id, device_display_name=current_device_display,
                            target_measurements_list=target_measurements, is_mkpred=is_mkpred,
                            operation_filter_mode=operation_filter_mode,
                            manual_thresholds=manual_thresholds,
                            measurement_limits=measurement_limits,
                            health_kpi_weights=health_kpi_weights
                        )
                    else:  # Dispositivos de Refrigeração
                        load_measurements = st.multiselect("Medições de Carga (Gatilho)", cleaned_series_names,
                                                           default=["MA_01"] if "MA_01" in cleaned_series_names else [],
                                                           key=f"loads_{device_id}")
                        op_current = st.number_input("Corrente Mín. de Operação (A)", value=1.0, step=0.1,
                                                     key=f"op_current_{device_id}")
                        stab_delay = st.number_input("Atraso de Estabilização (s)", value=300, key=f"stab_{device_id}")
                        shut_delay = st.number_input("Atraso de Desligamento (s)", value=60, key=f"shut_{device_id}")
                        startup_duration = st.number_input("Duração da Análise de Partida (s)", value=60,
                                                           key=f"startup_duration_{device_id}")

                        with st.expander("⚙️ Configurações de KPI de Refrigeração"):
                            limit_mode_display = st.radio("Definição dos Limites de Performance",
                                                          ("Manualmente", "Automaticamente (baseado no período)"),
                                                          key=f"limit_mode_{device_id}")
                            refrigeration_limit_mode = 'manual' if limit_mode_display == "Manualmente" else 'auto'

                            if refrigeration_limit_mode == 'manual':
                                st.markdown("**Faixas Ideais de Performance (Manual)**")
                                for tm in target_measurements:
                                    st.write(f"**{tm}**")
                                    col1, col2 = st.columns(2)
                                    min_val = col1.number_input("Mínimo Ideal", key=f"min_ref_{device_id}_{tm}",
                                                                format="%.2f")
                                    max_val = col2.number_input("Máximo Ideal", key=f"max_ref_{device_id}_{tm}",
                                                                format="%.2f", value=min_val + 1.0)
                                    if max_val > min_val:
                                        refrigeration_kpi_limits[tm] = {'min': min_val, 'max': max_val}
                            else:
                                st.info(
                                    "Os limites de Mínimo e Máximo Ideal serão calculados automaticamente (percentis 10 e 90) com base nos dados do período selecionado.")

                            acceptable_variation_percent = st.slider("Variação Máxima Aceitável (%)", 0, 100, 10, 1,
                                                                     key=f"var_perc_{device_id}")

                            st.markdown("**Pesos do Índice de Saúde**")
                            w_avail = st.slider("Peso da Disponibilidade", 0, 100, 50, 5, key=f"w_avail_{device_id}")
                            w_stab = st.slider("Peso da Estabilidade", 0, 100, 30, 5, key=f"w_stab_{device_id}")
                            w_perf = st.slider("Peso da Performance", 0, 100, 20, 5, key=f"w_perf_{device_id}")

                            total_weight = w_avail + w_stab + w_perf
                            if total_weight > 0:
                                refrigeration_kpi_weights['availability'] = w_avail / total_weight
                                refrigeration_kpi_weights['stability'] = w_stab / total_weight
                                refrigeration_kpi_weights['performance'] = w_perf / total_weight

                            st.caption(f"Disponibilidade: {refrigeration_kpi_weights['availability']:.0%} | "
                                       f"Estabilidade: {refrigeration_kpi_weights['stability']:.0%} | "
                                       f"Performance: {refrigeration_kpi_weights['performance']:.0%}")

                        device_config = DeviceAnalysisConfig(
                            device_id=device_id, device_display_name=current_device_display,
                            target_measurements_list=target_measurements, is_mkpred=is_mkpred,
                            load_measurement_names=load_measurements, operating_current=op_current,
                            stabilization_delay=stab_delay, shutdown_delay=shut_delay,
                            startup_duration=startup_duration,
                            refrigeration_kpi_limits=refrigeration_kpi_limits,
                            refrigeration_kpi_weights=refrigeration_kpi_weights,
                            acceptable_variation_percent=acceptable_variation_percent,
                            refrigeration_limit_mode=refrigeration_limit_mode
                        )
                    all_device_configs[device_id] = device_config

        st.markdown("---")
        fetch_alarms = st.checkbox("Buscar alarmes no período", value=True)

        if st.button("▶️ Iniciar Análise", type="primary", use_container_width=True):
            jobs_to_run: List[AnalysisJob] = []
            connection_config = ConnectionConfig(tenant_url=tenant, username=username, password=password)
            st.session_state.params = {'analysis_mode': analysis_mode}

            for device_id, config in all_device_configs.items():
                jobs_to_run.append(AnalysisJob(connection=connection_config, device_config=config,
                                               date_from=date_from.strftime('%Y-%m-%d'),
                                               date_to=date_to.strftime('%Y-%m-%d'),
                                               job_label='main', fetch_alarms=fetch_alarms, fetch_events=False))

            if jobs_to_run:
                st.session_state.jobs = jobs_to_run
                st.session_state.is_running = True
                st.session_state.log_messages = []
                st.session_state.results_df = None
                st.session_state.raw_data = None
                st.rerun()
            else:
                st.warning("Nenhuma análise para iniciar. Verifique as configurações.")


def display_trend_calculation_details(measurement_name, trend_indicators):
    """Mostra um painel com os detalhes do cálculo de tendência."""
    with st.container(border=True):
        st.subheader(f"Memória de Cálculo para: {measurement_name}")

        # Extrai os valores para facilitar a leitura
        first_value = trend_indicators.get('first_value', 0)
        last_value = trend_indicators.get('last_value', 0)
        mean_value = trend_indicators.get('mean', 0)
        slope = trend_indicators.get('slope', 0)
        r_squared = trend_indicators.get('r_squared', 0)
        growth_in_period = trend_indicators.get('growth_in_period', 0)
        rate_of_change_day = trend_indicators.get('rate_of_change_day', 0)
        growth_rate_period = trend_indicators.get('growth_rate_period', 0)

        col1, col2, col3 = st.columns(3)
        col1.metric("Valor Inicial", f"{first_value:.4f}")
        col2.metric("Valor Final", f"{last_value:.4f}")
        col3.metric("Média no Período", f"{mean_value:.4f}")

        st.markdown("---")
        st.markdown("##### KPIs Calculados")

        st.markdown("**Crescimento no Período (Real) (%)**")
        st.latex(r'''
        \text{Crescimento \%} = \frac{\text{Valor Final} - \text{Valor Inicial}}{|\text{Valor Inicial}|} \times 100
        ''')
        if abs(first_value) > 1e-9:
            st.markdown(
                f"= `({last_value:.4f} - {first_value:.4f}) / |{first_value:.4f}| * 100` = **`{growth_in_period:.2f}%`**")
        else:
            st.warning("O valor inicial é zero, não é possível calcular o crescimento percentual.")

        st.markdown("---")

        st.markdown("**Crescimento Médio no Período (Tendência) (%)**")
        st.latex(r'''
        \text{Cresc. Médio Período \%} = \text{Cresc. Diário \%} \times \text{Nº de Dias}
        ''')
        date_from_str = st.session_state.jobs[0].date_from
        date_to_str = st.session_state.jobs[0].date_to
        date_from_obj = datetime.strptime(date_from_str, '%Y-%m-%d')
        date_to_obj = datetime.strptime(date_to_str, '%Y-%m-%d')
        period_duration_days = (date_to_obj - date_from_obj).days + 1
        st.markdown(f"= `{rate_of_change_day:.2f}% \times {period_duration_days}` = **`{growth_rate_period:.2f}%`**")

        st.metric("Confiança da Tendência (R²)", f"{r_squared:.2%}")


def render_device_tab(current_device, main_job_label, is_report_mode=False):
    """Renderiza o conteúdo completo para a aba de um único dispositivo."""
    device_df = st.session_state.results_df[st.session_state.results_df['Dispositivo'] == current_device]
    kpis = st.session_state.kpis.get(main_job_label, {}).get(current_device, {})

    if not is_report_mode:
        all_components = ["Resumo dos Indicadores Chave", "KPIs Detalhados", "Análise Estatística",
                          "Visualizações de Dados"]
        with st.expander("⚙️ Personalizar Visualização"):
            selected_components = st.multiselect("Selecione os painéis para exibir:", options=all_components,
                                                 default=all_components, key=f"view_select_{current_device}")
    else:
        selected_components = ["Resumo dos Indicadores Chave", "KPIs Detalhados", "Análise Estatística",
                               "Visualizações de Dados"]

    if "Resumo dos Indicadores Chave" in selected_components:
        st.subheader("Resumo dos Indicadores Chave")
        if kpis.get('is_mkpred'):
            trend_data = st.session_state.trend_analysis.get(main_job_label, {}).get(current_device, {})
            health_indexes = [v['health_index'] for v in trend_data.values() if 'health_index' in v]
            critical_health_score = np.min(health_indexes) if health_indexes else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Índice de Saúde Preditivo", f"{critical_health_score:.1f}")
            col2.metric("Medições em Alerta",
                        len([s for s in trend_data.values() if s.get('status', '').startswith('🔴')]))
            col3.metric("Medições em Atenção",
                        len([s for s in trend_data.values() if s.get('status', '').startswith('🟡')]))
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric("Índice de Saúde", f"{kpis.get('health_index', 0):.1f}")
            col2.metric("Disponibilidade", f"{kpis.get('availability', 100):.2f}%")
            col3.metric("Nº de Paragens por Falha", kpis.get('number_of_faults', 0))
        st.markdown("---")

    if "KPIs Detalhados" in selected_components:
        st.subheader("KPIs Detalhados de Confiabilidade e Operação")
        if not kpis.get('is_mkpred'):
            kpi_cols1 = st.columns(3)
            kpi_cols1[0].metric("Tempo Parado por Falha", format_uptime(kpis.get('downtime_due_to_fault', 0)))
            kpi_cols1[1].metric("Número de Ciclos", kpis.get('num_cycles', 0))
            kpi_cols1[2].metric("Fator de Carga", f"{kpis.get('duty_cycle', 0):.2f}%")

            kpi_cols2 = st.columns(3)
            kpi_cols2[0].metric("Tempo de Operação Total", format_uptime(kpis.get('total_uptime', 0)))
            kpi_cols2[1].metric("Duração Média do Ciclo", format_uptime(kpis.get('mean_cycle_duration', 0)))
            kpi_cols2[2].metric("Tempo Médio Entre Ciclos", format_uptime(kpis.get('mean_time_between_cycles', 0)))

            for key, value in kpis.get('mean_values', {}).items():
                if key.startswith("Relação de Compressão"):
                    st.metric(key, f"{value:.2f}")
        else:
            st.info("KPIs de operação não são aplicáveis para dispositivos de análise de tendência contínua (MKPRED).")
        st.markdown("---")

    if "Análise Estatística" in selected_components:
        st.subheader("Análise Estatística Completa")
        if kpis.get('is_mkpred'):
            st.subheader("Análise de Tendência")
            trend_data = st.session_state.trend_analysis.get(main_job_label, {}).get(current_device, {})
            trend_df_data = []
            for m, ind in trend_data.items():
                trend_df_data.append({
                    "Medição": m,
                    "Status": ind.get('status'),
                    "Saúde": ind.get('health_index'),
                    "Cresc. no Período (%)": ind.get('growth_in_period'),
                    "Cresc. Médio no Período (%)": ind.get('growth_rate_period'),
                    "Cresc. Diário Médio (%)": ind.get('rate_of_change_day'),
                    "R² (Tendência)": ind.get('r_squared'),
                    "Média": ind.get('mean'),
                    "Desvio Padrão": ind.get('std_dev')
                })

            if trend_df_data:
                column_order = [
                    "Status", "Saúde", "Cresc. no Período (%)", "Cresc. Médio no Período (%)",
                    "Cresc. Diário Médio (%)", "R² (Tendência)", "Média", "Desvio Padrão"
                ]
                trend_df = pd.DataFrame(trend_df_data).set_index("Medição")
                trend_df = trend_df[[col for col in column_order if col in trend_df.columns]]

                st.dataframe(trend_df.style.format({
                    "Saúde": "{:.1f}",
                    "Cresc. no Período (%)": "{:.2f}%",
                    "Cresc. Médio no Período (%)": "{:.2f}%",
                    "Cresc. Diário Médio (%)": "{:.2f}%",
                    "R² (Tendência)": "{:.2%}",
                    "Média": "{:.4f}",
                    "Desvio Padrão": "{:.4f}"
                }), use_container_width=True)

                # --- SECÇÃO DE VALIDAÇÃO VISUAL ---
                valid_trend_measurements = [m for m, ind in trend_data.items() if ind]
                if valid_trend_measurements:
                    selected_measurement = st.selectbox(
                        "Ver detalhes do cálculo para:",
                        options=valid_trend_measurements,
                        key=f"details_select_{current_device}"
                    )
                    if selected_measurement:
                        display_trend_calculation_details(selected_measurement, trend_data[selected_measurement])

            else:
                st.info("Não há dados de tendência para exibir para os filtros selecionados.")

        st.subheader("Análise Estatística por Medição")
        display_df = device_df.drop(columns=['Dispositivo', 'Período/Job']).set_index('Medição')
        st.dataframe(display_df.style.format(precision=2), use_container_width=True)
        st.markdown("---")

    if "Visualizações de Dados" in selected_components:
        if not is_report_mode:
            corr_suggs = st.session_state.correlation_suggestions.get(main_job_label, {}).get(current_device, [])
            if corr_suggs:
                sugg_text = "  |  ".join([f"**{s['pair']}** (r={s['value']:.2f})" for s in corr_suggs])
                st.info(f"💡 **Sugestão de Correlação:** {sugg_text}")

        st.subheader("Visualizações de Dados")
        valid_measurements = device_df[device_df['Ocorrências'] > 0]['Medição'].tolist()

        if is_report_mode:
            # No modo relatório, mostramos apenas o gráfico de série temporal principal
            if valid_measurements:
                fig_ts = go.Figure(
                    layout=go.Layout(template="plotly_white", title_text=f'Série Temporal para {current_device}'))
                for m_name in valid_measurements:
                    raw_points = st.session_state.raw_data.get(main_job_label, {}).get(current_device, {}).get(m_name,
                                                                                                               [])
                    if raw_points:
                        times, values = zip(*raw_points)
                        fig_ts.add_trace(
                            go.Scatter(x=list(times), y=list(values), mode='lines', name=m_name, opacity=0.7))
                st.plotly_chart(fig_ts, use_container_width=True, key=f"ts_chart_report_{current_device}")
        else:
            graph_tab_list = ["Série Temporal", "Histograma", "Correlação"]
            if not kpis.get('is_mkpred'):
                graph_tab_list.extend(["Assinatura de Ciclo", "Análise de Partida"])
            if st.session_state.alarm_analysis.get(main_job_label, {}).get(current_device):
                graph_tab_list.append("Análise de Alarmes")

            graph_tabs = st.tabs(graph_tab_list)
            tab_map = {name: tab for name, tab in zip(graph_tab_list, graph_tabs)}

            if "Série Temporal" in tab_map:
                with tab_map["Série Temporal"]:
                    if valid_measurements:
                        selected_ts = st.multiselect("Medições para Série Temporal", valid_measurements,
                                                     default=valid_measurements[:2], key=f"ts_select_{current_device}")
                        if selected_ts:
                            fig_ts = go.Figure(
                                layout=go.Layout(template="streamlit",
                                                 title_text=f'Série Temporal para {current_device}'))
                            for m_name in selected_ts:
                                raw_points = st.session_state.raw_data.get(main_job_label, {}).get(current_device,
                                                                                                   {}).get(
                                    m_name, [])
                                if raw_points:
                                    times, values = zip(*raw_points)
                                    fig_ts.add_trace(
                                        go.Scatter(x=list(times), y=list(values), mode='lines', name=m_name,
                                                   opacity=0.7))

                                    if kpis.get('is_mkpred'):
                                        trend_indicators = st.session_state.trend_analysis.get(main_job_label, {}).get(
                                            current_device, {}).get(m_name)
                                        if trend_indicators and len(times) > 1:
                                            slope = trend_indicators.get('slope', 0)
                                            intercept = trend_indicators.get('intercept', 0)
                                            numeric_time = [(t - times[0]).total_seconds() for t in times]
                                            trend_line_y = [slope * t + intercept for t in numeric_time]
                                            fig_ts.add_trace(go.Scatter(x=list(times), y=trend_line_y, mode='lines',
                                                                        name=f'Tendência {m_name}',
                                                                        line=dict(dash='dash')))
                            st.plotly_chart(fig_ts, use_container_width=True, key=f"ts_chart_{current_device}")

            if "Histograma" in tab_map:
                with tab_map["Histograma"]:
                    if valid_measurements:
                        selected_hist = st.selectbox("Medição para Histograma", valid_measurements,
                                                     key=f"hist_select_{current_device}")
                        if selected_hist:
                            raw_points = st.session_state.raw_data.get(main_job_label, {}).get(current_device, {}).get(
                                selected_hist, [])
                            if raw_points:
                                _, values = zip(*raw_points)
                                fig_hist = go.Figure(data=[go.Histogram(x=list(values))],
                                                     layout=go.Layout(template="streamlit",
                                                                      title_text=f'Histograma de {selected_hist}'))
                                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_chart_{current_device}")

            if "Correlação" in tab_map:
                with tab_map["Correlação"]:
                    if len(valid_measurements) >= 2:
                        col1, col2 = st.columns(2)
                        x_axis = col1.selectbox("Eixo X", valid_measurements, index=0, key=f"corr_x_{current_device}")
                        y_axis = col2.selectbox("Eixo Y", valid_measurements, index=1, key=f"corr_y_{current_device}")

                        x_points = st.session_state.raw_data.get(main_job_label, {}).get(current_device, {}).get(x_axis,
                                                                                                                 [])
                        y_points = st.session_state.raw_data.get(main_job_label, {}).get(current_device, {}).get(y_axis,
                                                                                                                 [])

                        if x_points and y_points:
                            df_x = pd.DataFrame(x_points, columns=['time', x_axis]).set_index('time')
                            df_y = pd.DataFrame(y_points, columns=['time', y_axis]).set_index('time')
                            df_corr = pd.concat([df_x, df_y], axis=1).interpolate(method='time').dropna()

                            if not df_corr.empty:
                                corr_coef = df_corr[x_axis].corr(df_corr[y_axis])
                                fig_corr = go.Figure(
                                    data=go.Scatter(x=df_corr[x_axis], y=df_corr[y_axis], mode='markers'),
                                    layout=go.Layout(template="streamlit",
                                                     title_text=f'Correlação (r={corr_coef:.2f})'))
                                st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_chart_{current_device}")

            if "Assinatura de Ciclo" in tab_map:
                with tab_map["Assinatura de Ciclo"]:
                    cycle_analysis_data = st.session_state.cycle_signature_analysis.get(main_job_label, {}).get(
                        current_device, {})
                    if not cycle_analysis_data:
                        st.warning("Não há dados de assinatura de ciclo.")
                    else:
                        motor_measurement = next(iter(cycle_analysis_data))
                        analysis = cycle_analysis_data[motor_measurement]
                        fig_sig = go.Figure(layout=go.Layout(template="streamlit",
                                                             title_text=f'Assinatura de Ciclo para {motor_measurement}'))
                        x_axis = list(analysis['mean'].keys())
                        mean_curve, upper_bound, lower_bound = list(analysis['mean'].values()), list(
                            analysis['upper_bound'].values()), list(analysis['lower_bound'].values())
                        fig_sig.add_trace(
                            go.Scatter(x=x_axis + x_axis[::-1], y=upper_bound + lower_bound[::-1], fill='toself',
                                       fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                       name='Faixa de Normalidade'))
                        fig_sig.add_trace(
                            go.Scatter(x=x_axis, y=mean_curve, line=dict(color='rgb(0,100,80)'),
                                       name='Assinatura Média'))
                        st.plotly_chart(fig_sig, use_container_width=True, key=f"sig_chart_{current_device}")

            if "Análise de Partida" in tab_map:
                with tab_map["Análise de Partida"]:
                    startup_data = st.session_state.startup_analysis.get(main_job_label, {}).get(current_device, {})
                    if not startup_data:
                        st.warning(
                            "Não há dados de análise de partida. Verifique se uma medição de carga (ex: MA_01) foi selecionada e se há dados no início dos ciclos.")
                    else:
                        motor_measurement = next(iter(startup_data))
                        analysis = startup_data[motor_measurement]
                        fig_startup = go.Figure(layout=go.Layout(template="streamlit",
                                                                 title_text=f'Análise de Partida para {motor_measurement}'))
                        x_axis = list(analysis['mean'].keys())
                        mean_curve = list(analysis['mean'].values())
                        std_dev = list(analysis['std'].values())
                        upper_bound = [m + s for m, s in zip(mean_curve, std_dev)]
                        lower_bound = [m - s for m, s in zip(mean_curve, std_dev)]
                        fig_startup.add_trace(
                            go.Scatter(x=x_axis + x_axis[::-1], y=upper_bound + lower_bound[::-1], fill='toself',
                                       fillcolor='rgba(0,100,80,0.2)', line=dict(color='rgba(255,255,255,0)'),
                                       name='Faixa de Normalidade'))
                        fig_startup.add_trace(
                            go.Scatter(x=x_axis, y=mean_curve, line=dict(color='rgb(0,100,80)'), name='Partida Média'))
                        st.plotly_chart(fig_startup, use_container_width=True, key=f"startup_chart_{current_device}")

            if "Análise de Alarmes" in tab_map:
                with tab_map["Análise de Alarmes"]:
                    alarm_data = st.session_state.alarm_analysis.get(main_job_label, {}).get(current_device, {})
                    if not alarm_data or 'ranking' not in alarm_data:
                        st.info("Nenhum alarme encontrado no período analisado.")
                    else:
                        st.subheader("Ranking de Alarmes Mais Frequentes")
                        df_ranking = pd.DataFrame(alarm_data['ranking'])
                        st.dataframe(df_ranking[['Alarme', 'Ocorrências', 'MTBA']], use_container_width=True)

                        st.subheader("Ocorrências por Tipo de Alarme")
                        df_by_type = pd.DataFrame(alarm_data['by_type'])
                        fig_alarm_type = go.Figure(data=[go.Bar(x=df_by_type['Tipo'], y=df_by_type['Ocorrências'])],
                                                   layout=go.Layout(template="streamlit",
                                                                    title_text="Contagem por Tipo de Alarme"))
                        st.plotly_chart(fig_alarm_type, use_container_width=True, key=f"alarm_chart_{current_device}")

    if is_report_mode:
        st.subheader("Diagnóstico e Recomendações")
        st.text_area("Insira os seus comentários aqui:", height=200, key=f"report_comments_{current_device}")


def display_results_area():
    """Renderiza a área principal de resultados."""
    if st.session_state.results_df is None:
        st.info("Configure e inicie uma análise usando o painel à esquerda.")
        return

    if st.session_state.results_df.empty:
        st.warning("Nenhum dado encontrado para os parâmetros selecionados.")
        return

    # --- Botão para entrar no Modo de Relatório ---
    if st.button("🖨️ Preparar Relatório para Exportação", use_container_width=True):
        st.session_state.report_mode = True
        st.rerun()

    st.success("Análise Concluída!")
    st.metric("Total de Chamadas à API", st.session_state.api_call_count)
    st.markdown("---")

    analysis_mode = st.session_state.params.get('analysis_mode', 'Análise Detalhada')

    if analysis_mode == "Comparar Dispositivos":
        st.header("📊 Análise Comparativa (Benchmarking)")
        main_job_label = next(iter(st.session_state.kpis.keys()), None)
        if main_job_label:
            kpis_data = st.session_state.kpis.get(main_job_label, {})
            if len(kpis_data) >= 2:
                df_kpis = pd.DataFrame.from_dict(kpis_data, orient='index')
                kpis_to_compare = {
                    'health_index': 'Índice de Saúde', 'availability': 'Disponibilidade',
                    'number_of_faults': 'Nº de Falhas', 'duty_cycle': 'Fator de Carga (%)',
                    'mean_cycle_duration': 'Duração Média Ciclo (s)', 'num_cycles': 'Nº de Ciclos'
                }
                df_compare = df_kpis[[k for k in kpis_to_compare.keys() if k in df_kpis.columns]].rename(
                    columns=kpis_to_compare)
                df_avg = df_compare.mean()
                df_dev = ((df_compare - df_avg) / df_avg * 100).replace([np.inf, -np.inf], 100).add_suffix(
                    ' (% Desvio)')

                def style_deviation_df(df):
                    styles = pd.DataFrame('', index=df.index, columns=df.columns)
                    higher_is_better = ['Índice de Saúde (% Desvio)', 'Disponibilidade (% Desvio)']
                    lower_is_better = ['Nº de Falhas (% Desvio)']
                    for col in df.columns:
                        for idx in df.index:
                            val = df.loc[idx, col]
                            if pd.isna(val) or not isinstance(val, (int, float)): continue
                            style = 'background-color: '
                            if val > 10:
                                if col in higher_is_better:
                                    styles.loc[idx, col] = style + '#3D9970'
                                elif col in lower_is_better:
                                    styles.loc[idx, col] = style + '#FF4136'
                                else:
                                    styles.loc[idx, col] = style + '#FF851B'
                            elif val < -10:
                                if col in higher_is_better:
                                    styles.loc[idx, col] = style + '#FF4136'
                                elif col in lower_is_better:
                                    styles.loc[idx, col] = style + '#3D9970'
                                else:
                                    styles.loc[idx, col] = style + '#FF851B'
                    return styles

                st.dataframe(df_dev.style.apply(style_deviation_df, axis=None).format("{:.1f}%"))
                st.markdown("---")

    st.header("🔍 Análise Detalhada por Dispositivo")
    main_job_label = next(iter(st.session_state.kpis.keys()), None)
    if main_job_label:
        analyzed_devices = list(st.session_state.kpis.get(main_job_label, {}).keys())
        if analyzed_devices:
            device_tabs = st.tabs(analyzed_devices)
            for i, tab in enumerate(device_tabs):
                with tab:
                    render_device_tab(analyzed_devices[i], main_job_label)


# --- FUNÇÃO PARA GERAR PDF (MODIFICADA PARA FASE 1) ---
def generate_pdf_report(report_config, selected_sections, _analyzed_devices, _main_job_label):
    """Gera o conteúdo HTML do relatório e o converte para PDF."""
    html_parts = []

    # --- Estilo CSS para o PDF ---
    report_css = """
    @page {
        size: A4;
        margin: 1.5cm;
    }
    @page:not(:first) {
        @top-center {
            content: element(header);
        }
        @bottom-center {
            content: element(footer);
        }
    }
    .header {
        position: running(header);
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 2px solid #0056b3;
        padding-bottom: 5px;
        width: 100%;
    }
    .footer {
        position: running(footer);
        text-align: center;
        font-size: 10px;
        color: #555;
        border-top: 1px solid #ccc;
        padding-top: 5px;
        width: 100%;
    }
    .logo { max-height: 50px; max-width: 150px; }
    body { font-family: 'Helvetica', sans-serif; color: #333; }
    h1, h2, h3 { color: #0056b3; }
    h1 { text-align: center; margin-bottom: 20px;}
    h2 { border-bottom: 2px solid #0056b3; padding-bottom: 5px; margin-top: 30px;}
    h3 { border-bottom: 1px solid #ccc; padding-bottom: 3px; margin-top: 20px;}
    table {
        width: 100%;
        border-collapse: collapse;
        margin-bottom: 20px;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th { background-color: #f2f2f2; }
    .page-break { page-break-before: always; }
    .cover-page {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        align-items: center;
        height: 25cm; /* Altura da área de conteúdo A4 */
        text-align: center;
    }
    .cover-title { font-size: 28px; margin-top: 2cm; }
    .cover-subtitle { font-size: 20px; color: #555; }
    .cover-info { margin-top: 4cm; }
    .cover-footer { width: 100%; }
    .signatures { margin-top: 3cm; display: flex; justify-content: space-around; width: 80%;}
    .signature-box { border-top: 1px solid #333; padding-top: 5px; width: 40%;}
    .comments {
        background-color: #eef;
        border-left: 5px solid #0056b3;
        padding: 15px;
        margin-top: 20px;
        white-space: pre-wrap;
    }
    """

    # --- Cabeçalho e Rodapé ---
    client_logo_html = ""
    if report_config.get('client_logo_bytes'):
        logo_base64 = base64.b64encode(report_config['client_logo_bytes']).decode()
        client_logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo">'

    my_logo_html = ""
    if report_config.get('my_logo_bytes'):
        logo_base64 = base64.b64encode(report_config['my_logo_bytes']).decode()
        my_logo_html = f'<img src="data:image/png;base64,{logo_base64}" class="logo">'

    header_html = f'<div class="header">{client_logo_html}{my_logo_html}</div>'
    footer_html = f'<div class="footer">Seu Nome de Empresa | Endereço | Telefone <br/> Página <span class="page-number"></span> de <span class="total-pages"></span></div>'

    html_parts.append(f"""
    <html>
        <head>
            <meta charset="UTF-8">
            <style>{report_css}</style>
        </head>
        <body>
            {header_html}
            {footer_html}

            <!-- Página de Rosto -->
            <div class="cover-page">
                <div>
                    <h1 class="cover-title">Relatório de Monitoramento Online</h1>
                    <p class="cover-subtitle">{report_config.get('client_name', 'Cliente não especificado')}</p>
                </div>
                <div class="cover-info">
                    <p><strong>Período da Análise:</strong> {st.session_state.jobs[0].date_from} a {st.session_state.jobs[0].date_to}</p>
                    <p><strong>Data de Emissão:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                </div>
                <div class="cover-footer">
                    <div class="signatures">
                        <div class="signature-box">
                            <p>{report_config.get('prepared_by', '_________________________')}</p>
                            <p><strong>Elaborado por</strong></p>
                        </div>
                        <div class="signature-box">
                            <p>{report_config.get('approved_by', '_________________________')}</p>
                            <p><strong>Aprovado por</strong></p>
                        </div>
                    </div>
                </div>
            </div>
    """)

    # --- Conteúdo Principal (começa na página seguinte) ---
    for i, device in enumerate(_analyzed_devices):
        html_parts.append('<div class="page-break"></div>')
        html_parts.append(f"<h2>Análise do Dispositivo: {device}</h2>")

        kpis = st.session_state.kpis.get(_main_job_label, {}).get(device, {})
        device_df = st.session_state.results_df[st.session_state.results_df['Dispositivo'] == device]

        # --- Tabela de KPIs ---
        if "Tabela de KPIs" in selected_sections:
            html_parts.append("<h3>Indicadores Chave de Performance (KPIs)</h3>")
            html_parts.append(generate_kpi_table_html(kpis))

        # --- Tabela de Análise Estatística ---
        if "Análise Estatística" in selected_sections:
            html_parts.append("<h3>Análise Estatística por Medição</h3>")
            display_df = device_df.drop(columns=['Dispositivo', 'Período/Job']).set_index('Medição')
            html_parts.append(display_df.to_html(classes='nice-table', float_format='{:.2f}'.format))

        # --- Gráficos ---
        if "Gráficos" in selected_sections:
            html_parts.append("<h3>Visualizações de Dados</h3>")
            valid_measurements = device_df[device_df['Ocorrências'] > 0]['Medição'].tolist()
            if valid_measurements:
                fig_ts = go.Figure(
                    layout=go.Layout(template="plotly_white", title_text=f'Série Temporal para {device}'))
                for m_name in valid_measurements:
                    raw_points = st.session_state.raw_data.get(_main_job_label, {}).get(device, {}).get(m_name, [])
                    if raw_points:
                        times, values = zip(*raw_points)
                        fig_ts.add_trace(
                            go.Scatter(x=list(times), y=list(values), mode='lines', name=m_name, opacity=0.7))
                img_bytes = fig_ts.to_image(format="png", width=800, height=400, scale=2)
                img_base64 = base64.b64encode(img_bytes).decode()
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" style="width: 100%;">')

        # --- Comentários ---
        comments = st.session_state.get(f"report_comments_{device}", "Nenhum comentário adicionado.")
        html_parts.append("<h3>Diagnóstico e Recomendações</h3>")
        html_parts.append(f'<div class="comments">{comments}</div>')

    html_parts.append("</body></html>")
    full_html = "".join(html_parts)

    # --- Geração do PDF ---
    pdf_bytes = HTML(string=full_html).write_pdf()
    return pdf_bytes


def generate_kpi_table_html(kpis):
    """Gera uma tabela HTML para os KPIs com cores baseadas em limiares."""

    def get_class(value, thresholds):
        if value >= thresholds[0]: return "good"
        if value >= thresholds[1]: return "warning"
        return "critical"

    health_class = get_class(kpis.get('health_index', 0), [80, 60])
    avail_class = get_class(kpis.get('availability', 100), [95, 90])

    # Lógica invertida para falhas: menos é melhor
    num_faults = kpis.get('number_of_faults', 0)
    if num_faults <= 1:
        faults_class = "good"
    elif num_faults <= 5:
        faults_class = "warning"
    else:
        faults_class = "critical"

    kpi_data = {
        "Indicador": ["Índice de Saúde", "Disponibilidade", "Nº de Paragens por Falha", "Fator de Carga"],
        "Valor": [f"{kpis.get('health_index', 0):.1f}", f"{kpis.get('availability', 100):.2f}%", f"{num_faults}",
                  f"{kpis.get('duty_cycle', 0):.2f}%"],
        "Classe": [health_class, avail_class, faults_class, ""]
    }
    df = pd.DataFrame(kpi_data)

    html = '<style>.kpi-table td.good { background-color: #d4edda !important; } .kpi-table td.warning { background-color: #fff3cd !important; } .kpi-table td.critical { background-color: #f8d7da !important; }</style>'
    html += '<table class="kpi-table"><tr><th>Indicador</th><th>Valor</th></tr>'
    for _, row in df.iterrows():
        html += f'<tr><td>{row["Indicador"]}</td><td class="{row["Classe"]}">{row["Valor"]}</td></tr>'
    html += '</table>'
    return html


def display_report_mode():
    """Renderiza a visualização de relatório para impressão."""
    st.markdown('<div class="report-mode">', unsafe_allow_html=True)  # Ativa o CSS para esconder a sidebar

    st.title("Preparação do Relatório")

    if st.button("⬅️ Voltar ao Dashboard", key="back_to_dash"):
        st.session_state.report_mode = False
        st.session_state.pdf_generated = False
        if 'pdf_for_download' in st.session_state:
            del st.session_state.pdf_for_download
        st.rerun()

    st.markdown("---")

    # --- OPÇÕES DE PERSONALIZAÇÃO DA FASE 1 ---
    st.header("1. Personalização do Relatório")

    client_name = st.text_input("Nome do Cliente", key="client_name")

    col1, col2 = st.columns(2)
    with col1:
        client_logo_file = st.file_uploader("Carregue o logótipo do Cliente", type=['png', 'jpg', 'jpeg'],
                                            key="client_logo")
    with col2:
        my_logo_file = st.file_uploader("Carregue o seu Logótipo", type=['png', 'jpg', 'jpeg'], key="my_logo")

    col3, col4 = st.columns(2)
    with col3:
        prepared_by = st.text_input("Elaborado por:", key="prepared_by")
    with col4:
        approved_by = st.text_input("Aprovado por:", key="approved_by")

    executive_summary = st.text_area("Resumo Executivo (Opcional)", key="executive_summary", height=150,
                                     help="Escreva um parágrafo de introdução que aparecerá no início do relatório.")

    all_sections = ["Tabela de KPIs", "Análise Estatística", "Gráficos", "Diagnóstico e Recomendações"]
    selected_sections = st.multiselect("Selecione as secções para incluir no relatório:", all_sections,
                                       default=all_sections)

    st.markdown("---")

    st.header("2. Comentários por Dispositivo")
    main_job_label = next(iter(st.session_state.kpis.keys()), None)
    if main_job_label:
        analyzed_devices = list(st.session_state.kpis.get(main_job_label, {}).keys())
        for device in analyzed_devices:
            with st.expander(f"Adicionar comentários para: {device}"):
                st.text_area("Diagnóstico e Recomendações:", height=200, key=f"report_comments_{device}")

        st.markdown("---")
        st.header("3. Gerar e Fazer Download")

        if st.button("Gerar Relatório PDF", key="generate_pdf"):
            with st.spinner("Gerando o seu relatório em PDF..."):
                report_config = {
                    'client_logo_bytes': client_logo_file.getvalue() if client_logo_file else None,
                    'my_logo_bytes': my_logo_file.getvalue() if my_logo_file else None,
                    'client_name': client_name,
                    'prepared_by': prepared_by,
                    'approved_by': approved_by,
                    'executive_summary': executive_summary
                }
                pdf_file = generate_pdf_report(report_config, selected_sections, analyzed_devices, main_job_label)
                st.session_state.pdf_for_download = pdf_file
                st.session_state.pdf_generated = True

        if st.session_state.get("pdf_generated", False):
            st.download_button(
                label="📥 Fazer Download do Relatório",
                data=st.session_state.pdf_for_download,
                file_name=f"relatorio_analise_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )

    st.markdown('</div>', unsafe_allow_html=True)


# --- Inicialização do Estado da Sessão ---
if 'is_running' not in st.session_state: st.session_state.is_running = False
if 'log_messages' not in st.session_state: st.session_state.log_messages = []
if 'status_text' not in st.session_state: st.session_state.status_text = "Aguardando início..."
if 'progress_value' not in st.session_state: st.session_state.progress_value = 0.0
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'raw_data' not in st.session_state: st.session_state.raw_data = {}
if 'log_queue' not in st.session_state: st.session_state.log_queue = Queue()
if 'api_call_count' not in st.session_state: st.session_state.api_call_count = 0
if 'kpis' not in st.session_state: st.session_state.kpis = {}
if 'alarms_events' not in st.session_state: st.session_state.alarms_events = {}
if 'alarm_analysis' not in st.session_state: st.session_state.alarm_analysis = {}
if 'startup_analysis' not in st.session_state: st.session_state.startup_analysis = {}
if 'trend_analysis' not in st.session_state: st.session_state.trend_analysis = {}
if 'cycle_signature_analysis' not in st.session_state: st.session_state.cycle_signature_analysis = {}
if 'correlation_suggestions' not in st.session_state: st.session_state.correlation_suggestions = {}
if 'params' not in st.session_state: st.session_state.params = {}
if 'report_mode' not in st.session_state: st.session_state.report_mode = False
if 'pdf_generated' not in st.session_state: st.session_state.pdf_generated = False

# --- Corpo Principal da Aplicação ---
if st.session_state.report_mode:
    display_report_mode()
else:
    st.title("📊 Analisador de Performance de Ativos")
    display_configuration_sidebar()

    if st.session_state.is_running:
        if 'analysis_thread' not in st.session_state or not st.session_state.analysis_thread.is_alive():
            stop_event = Event()
            st.session_state.stop_event = stop_event
            st.session_state.analysis_thread = Thread(target=perform_analysis_master_thread, args=(
                stop_event, st.session_state.log_queue, st.session_state.jobs))
            st.session_state.analysis_thread.start()

        while not st.session_state.log_queue.empty():
            msg = st.session_state.log_queue.get()
            if msg['type'] == 'log':
                st.session_state.log_messages.append(msg)
            elif msg['type'] == 'status':
                st.session_state.status_text = msg['data']
                if 'progress' in msg: st.session_state.progress_value = msg['progress']
            elif msg['type'] == 'finished':
                st.session_state.is_running = False
                data = msg['data']
                st.session_state.api_call_count = data['api_calls']
                st.session_state.kpis = data['kpis']
                st.session_state.alarms_events = data['alarms_events']
                st.session_state.alarm_analysis = data['alarm_analysis']
                st.session_state.startup_analysis = data['startup_analysis']
                st.session_state.trend_analysis = data['trend_analysis']
                st.session_state.cycle_signature_analysis = data['cycle_signature_analysis']
                st.session_state.raw_data = data['raw']
                st.session_state.correlation_suggestions = data['correlation_suggestions']

                df_data = []
                for job_label, devices in data['results'].items():
                    for device_name, results in devices.items():
                        for name, res_data in results.items():
                            df_data.append({
                                "Período/Job": job_label, "Dispositivo": device_name, "Medição": name,
                                "Mínimo": res_data.get('min'), "Máximo": res_data.get('max'),
                                "Amplitude": res_data.get('range'),
                                "Média": res_data.get('mean'), "Mediana": res_data.get('median'),
                                "Desvio Padrão": res_data.get('std_dev'), "P95": res_data.get('p95'),
                                "Timestamp Mínimo": format_timestamp_to_brasilia(res_data.get('min_time')),
                                "Timestamp Máximo": format_timestamp_to_brasilia(res_data.get('max_time')),
                                "Ocorrências": res_data.get('count_valid')
                            })
            st.session_state.results_df = pd.DataFrame(df_data) if df_data else pd.DataFrame(
                columns=["Dispositivo"])
            st.rerun()

    st.info(st.session_state.status_text)
    st.progress(st.session_state.progress_value)

    st.markdown("### Log de Execução")
    log_html = "".join([f'<div class="log-entry log-{msg.get("color", "")}">{msg["data"]}</div>' for msg in
                        st.session_state.log_messages])
    st.markdown(f'<div class="log-container">{log_html}</div>', unsafe_allow_html=True)

    if st.button("Cancelar Análise", type="primary"):
        st.session_state.stop_event.set()
        st.info("Cancelamento solicitado. Aguardando a finalização do ciclo atual...")

    time.sleep(1)
    st.rerun()
else:
    display_results_area()

