
# -*- coding: utf-8 -*-
"""
Streamlit UI for purchase vs renting simulations with SQLite run logging.

Notes
-----
- Variables in English.
- Comments in Spanish.
- Docstrings follow NumPy style.
"""
from __future__ import annotations
import json
import sqlite3
from datetime import datetime
from dataclasses import asdict, is_dataclass
from typing import Dict, Any

import streamlit as st
import pandas as pd

# Importar el módulo existente con los cálculos
from vehicle_capex_opex import (
    VehicleSpec, MaintenanceSpec, InsuranceSpec, TaxesSpec, UsageSpec,
    PurchaseFinanceSpec, DepreciationSpec, RentingSpec,
    one_year_operating_cost, simulate_purchase_over_horizon, simulate_renting_over_horizon,
    compare_purchase_vs_renting
)

# ===============================
# Utilidades
# ===============================

def dataclass_to_dict(obj):
    """Convertir dataclass anidado a dict (recursivo)."""
    if is_dataclass(obj):
        return {k: dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_to_dict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: dataclass_to_dict(v) for k, v in obj.items()}
    return obj

def init_db(db_path: str = "runs.db") -> None:
    """Crear la tabla de registros si no existe."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scenario TEXT NOT NULL,
                horizon_years INTEGER NOT NULL,
                params_json TEXT NOT NULL,
                result_json TEXT NOT NULL
            );
            """
        )
        conn.commit()

def log_run(scenario: str, years: int, params: Dict[str, Any], result: Dict[str, Any], db_path: str = "runs.db") -> None:
    """Guardar un registro de la corrida con parámetros y resultados en JSON."""
    ts = datetime.utcnow().isoformat()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO runs (timestamp, scenario, horizon_years, params_json, result_json) VALUES (?, ?, ?, ?, ?)",
            (ts, scenario, years, json.dumps(params, ensure_ascii=False), json.dumps(result, ensure_ascii=False))
        )
        conn.commit()

def load_runs(db_path: str = "runs.db") -> pd.DataFrame:
    """Cargar el historial de corridas."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM runs ORDER BY id DESC", conn)
    return df

# ===============================
# Interfaz de Streamlit
# ===============================

st.set_page_config(page_title="Compra vs Renting - Simulador", layout="wide")
st.title("🚗 Simulador Compra vs Renting")
st.caption("Mantiene la lógica de **vehicle_capex_opex.py** y añade UI para introducir parámetros y registrar cada ejecución en SQLite.")

# Inicializar DB
init_db()

with st.sidebar:
    st.header("Escenario")
    scenario = st.text_input("Nombre del escenario", value="Mi Escenario")
    years = st.number_input("Horizonte (años)", min_value=1, max_value=15, value=3, step=1)

    st.divider()
    st.subheader("Vehículo")
    engine_cc = st.number_input("Cilindrada (cc)", min_value=600, max_value=6000, value=999, step=1)
    cylinders = st.number_input("Cilindros", min_value=2, max_value=12, value=3, step=1)
    power_kw = st.number_input("Potencia (kW)", min_value=30.0, max_value=400.0, value=74.0, step=1.0)
    fuel_type = st.selectbox("Tipo de combustible", options=['gasoline', 'diesel', 'hybrid', 'electric'], index=2)

    st.subheader("Uso")
    age_years = st.number_input("Edad del vehículo (años)", min_value=0.0, max_value=20.0, value=0.0, step=1.0, format="%.1f")
    annual_km = st.number_input("Kilómetros anuales", min_value=0.0, max_value=60000.0, value=15000.0, step=1000.0)
    odometer_km = st.number_input("Odómetro actual (km)", min_value=0.0, max_value=400000.0, value=0.0, step=1000.0)

    st.divider()
    st.subheader("Financiación de Compra")
    finance_mode = st.radio("Modo de financiación", options=['amortized', 'custom'], horizontal=True, index=1)
    price = st.number_input("Precio (€)", min_value=0.0, value=14793.0, step=100.0)
    down_payment = st.number_input("Entrada / Down payment (€)", min_value=0.0, value=2500.0, step=100.0)

    if finance_mode == 'amortized':
        loan_rate_annual = st.number_input("Interés anual (ej. 0.06)", min_value=0.0, max_value=1.0, value=0.06, step=0.005, format="%.3f")
        loan_term_months = st.number_input("Plazo del préstamo (meses)", min_value=0, max_value=120, value=60, step=1)
        installment_amount = 0.0
        installment_count = 0
        balloon_final = 0.0
    else:
        loan_rate_annual = 0.0
        loan_term_months = 0
        installment_amount = st.number_input("Cuota mensual fija (€)", min_value=0.0, value=119.0, step=1.0)
        installment_count = st.number_input("Número de cuotas", min_value=0, max_value=182, value=36, step=1)
        balloon_final = st.number_input("Cuota final / Balloon (€)", min_value=0.0, value=10164, step=50.0)

    st.divider()
    st.subheader("Depreciación")
    dep_method = st.radio("Método", options=['declining', 'straight'], horizontal=True, index=0)
    if dep_method == 'declining':
        declining_rate = st.number_input("Tasa anual de depreciación", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
        floor_fraction = st.number_input("Fracción de piso (mínimo del valor original)", min_value=0.0, max_value=1.0, value=0.50, step=0.01, format="%.2f")
        life_years = 5
        salvage_fraction = 0.26
    else:
        life_years = st.number_input("Vida útil (años)", min_value=1, max_value=20, value=5, step=1)
        salvage_fraction = st.number_input("Fracción de salvamento", min_value=0.0, max_value=1.0, value=0.26, step=0.01, format="%.2f")
        declining_rate = 0.10
        floor_fraction = 0.26

    st.divider()
    st.subheader("Renting")
    monthly_fee = st.number_input("Cuota mensual (€)", min_value=0.0, value=280, step=5.0)
    renting_months = st.number_input("Meses de contrato", min_value=1, max_value=120, value=36, step=1)
    upfront_fee = st.number_input("Cuota inicial (€)", min_value=0.0, value=0.0, step=50.0)

    includes_insurance = st.checkbox("Incluye seguro", value=True)
    includes_maintenance = st.checkbox("Incluye mantenimiento", value=True)
    includes_tyres = st.checkbox("Incluye neumáticos", value=True)
    includes_taxes = st.checkbox("Incluye impuestos", value=True)

    annual_km_allowance = st.number_input("Km anuales incluidos", min_value=0.0, max_value=200000.0, value=15000.0, step=1000.0)
    excess_km_cost = st.number_input("Costo por km excedente (€)", min_value=0.0, max_value=10.0, value=0.06, step=0.01)

# Botón de ejecución
run_button = st.button("▶️ Ejecutar simulación", type="primary", use_container_width=True)

# Crear objetos de especificación a partir de los inputs
vehicle = VehicleSpec(engine_cc=engine_cc, cylinders=cylinders, power_kw=power_kw, fuel_type=fuel_type)
usage = UsageSpec(age_years=age_years, annual_km=annual_km, odometer_km=odometer_km)
maint = MaintenanceSpec()  # usar valores por defecto definidos en el módulo
ins = InsuranceSpec()      # base por defecto; si necesitas editar, puedes extender UI
taxes = TaxesSpec()        # tramos por defecto; editable si se desea en el futuro

finance = PurchaseFinanceSpec(
    mode=finance_mode, price=price, down_payment=down_payment,
    loan_rate_annual=loan_rate_annual, loan_term_months=int(loan_term_months),
    installment_amount=installment_amount, installment_count=int(installment_count), balloon_final=balloon_final
)

depreciation = DepreciationSpec(
    method=dep_method, declining_rate=declining_rate, floor_fraction=floor_fraction,
    life_years=int(life_years), salvage_fraction=salvage_fraction
)

renting = RentingSpec(
    monthly_fee=monthly_fee, months=int(renting_months), upfront_fee=upfront_fee,
    includes_insurance=includes_insurance, includes_maintenance=includes_maintenance,
    includes_tyres=includes_tyres, includes_taxes=includes_taxes,
    annual_km_allowance=annual_km_allowance, excess_km_cost=excess_km_cost
)

# Ejecutar simulaciones
if run_button:
    # Calcular compra y renting por separado
    purchase_res = simulate_purchase_over_horizon(
        years=years, vehicle=vehicle, maint=maint, ins=ins, taxes=taxes,
        usage=usage, finance=finance, depreciation=depreciation
    )
    renting_res = simulate_renting_over_horizon(
        years=years, renting=renting, vehicle=vehicle, maint=maint, ins=ins, taxes=taxes, usage=usage
    )
    compare_res = compare_purchase_vs_renting(
        scenario, years, vehicle, maint, ins, taxes, usage, finance, depreciation, renting
    )

    # Mostrar resultados
    st.success("Simulación completada.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Compra - Resumen")
        st.dataframe(pd.DataFrame([purchase_res]))

        st.subheader("Coste Operativo Estimado (Año 1)")
        y1 = one_year_operating_cost(vehicle, maint, ins, taxes, usage)
        st.dataframe(pd.DataFrame([y1]))

    with col2:
        st.subheader("Renting - Resumen")
        st.dataframe(pd.DataFrame([renting_res]))

        st.subheader("Comparativa")
        st.dataframe(pd.DataFrame([compare_res]))

    # Guardar registro en SQLite
    params_payload = {
        "scenario": scenario,
        "years": years,
        "vehicle": dataclass_to_dict(vehicle),
        "usage": dataclass_to_dict(usage),
        "maintenance": dataclass_to_dict(maint),
        "insurance": dataclass_to_dict(ins),
        "taxes": dataclass_to_dict(taxes),
        "finance": dataclass_to_dict(finance),
        "depreciation": dataclass_to_dict(depreciation),
        "renting": dataclass_to_dict(renting),
    }
    result_payload = {
        "purchase": purchase_res,
        "renting": renting_res,
        "compare": compare_res,
    }
    log_run(scenario, int(years), params_payload, result_payload)

    st.info("La ejecución ha sido registrada en SQLite (runs.db).")

st.divider()
st.header("📜 Historial")
df_runs = load_runs()
st.dataframe(df_runs, use_container_width=True)
if not df_runs.empty:
    # Permitir descargar el historial completo como CSV
    csv = df_runs.to_csv(index=False).encode('utf-8')
    st.download_button("Descargar historial (CSV)", data=csv, file_name="runs_history.csv", mime="text/csv")
