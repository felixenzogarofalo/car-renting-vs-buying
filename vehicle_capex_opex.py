# -*- coding: utf-8 -*-
"""
Programa de simulación de costes operativos y comparativa Compra vs Renting.

Notas de estilo para el usuario:
- Variables en inglés.
- Comentarios en español.
- Docstrings con el estándar de NumPy.

Este módulo permite:
1) Calcular costes operativos anuales (mantenimiento preventivo/correctivo, neumáticos, seguro, impuestos).
2) Modelar IVTM a partir de la potencia fiscal (CVF).
3) Incluir odómetro (km acumulados) como factor clave de mantenimiento.
4) Simular una compra con financiación: pagos, saldo pendiente y valor devaluado (depreciación).
   - Modo 'amortized' (cuota calculada con interés y plazo)
   - Modo 'custom' (número de cuotas fijo, monto fijo y cuota final)
5) Simular renting (con o sin cuota inicial) y comparar contra compra.
6) Correr múltiples escenarios y ver resultados en tabla.

Requiere: numpy, pandas
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional
import math
import numpy as np
import pandas as pd


# ===============================
# Utilidades financieras y de ayuda
# ===============================

def annuity_payment(principal: float, annual_rate: float, term_months: int) -> float:
    """
    Calcular la cuota mensual de un préstamo (sistema francés).
    
    Parameters
    ----------
    principal : float
        Principal del préstamo.
    annual_rate : float
        Tasa de interés nominal anual (por ejemplo 0.06 = 6%).
    term_months : int
        Plazo total en meses.
    
    Returns
    -------
    float
        Pago mensual (cuota).
    """
    if term_months <= 0:
        return 0.0
    # Si la tasa anual es 0, cuota simple
    if annual_rate == 0:
        return principal / term_months
    r = annual_rate / 12.0
    return principal * (r / (1 - (1 + r) ** (-term_months)))


def loan_balance(principal: float, annual_rate: float, term_months: int, months_elapsed: int) -> float:
    """
    Saldo pendiente de un préstamo tras cierto número de meses.
    
    Parameters
    ----------
    principal : float
        Principal del préstamo.
    annual_rate : float
        Tasa de interés nominal anual.
    term_months : int
        Plazo total (meses).
    months_elapsed : int
        Meses transcurridos.
    
    Returns
    -------
    float
        Saldo remanente del préstamo.
    """
    if months_elapsed <= 0:
        return principal
    if months_elapsed >= term_months:
        return 0.0
    pmt = annuity_payment(principal, annual_rate, term_months)
    r = annual_rate / 12.0
    if annual_rate == 0:
        return max(principal - pmt * months_elapsed, 0.0)
    return principal * (1 + r) ** months_elapsed - pmt * (((1 + r) ** months_elapsed - 1) / r)


def declining_balance_value(purchase_price: float, years: float, annual_rate: float, floor_fraction: float = 0.1) -> float:
    """
    Valor residual por depreciación de saldo decreciente.
    
    Parameters
    ----------
    purchase_price : float
        Precio de compra.
    years : float
        Años transcurridos.
    annual_rate : float
        Tasa de depreciación anual (e.g. 0.2 = 20%/año).
    floor_fraction : float, optional
        Fracción mínima del precio original que el valor no deberá traspasar.
    
    Returns
    -------
    float
        Valor residual estimado.
    """
    value = purchase_price * ((1 - annual_rate) ** years)
    floor_val = purchase_price * floor_fraction
    return max(value, floor_val)


def straight_line_value(purchase_price: float, years: float, life_years: float, salvage_fraction: float = 0.1) -> float:
    """
    Valor residual por depreciación lineal.
    
    Parameters
    ----------
    purchase_price : float
        Precio de compra.
    years : float
        Años transcurridos.
    life_years : float
        Vida útil (años) para llevar el activo al valor de salvamento.
    salvage_fraction : float
        Fracción del precio original como valor de salvamento.
    
    Returns
    -------
    float
        Valor residual estimado.
    """
    salvage_value = purchase_price * (salvage_fraction)
    annual_dep = (purchase_price - salvage_value) / life_years
    value = purchase_price - annual_dep * years
    return max(value, salvage_value)


# ===============================
# Modelo técnico: IVTM, ITV, mantenimiento, seguro, etc.
# ===============================

def compute_cvf(engine_cc: float, cylinders: int) -> float:
    """
    Calcular la potencia fiscal (CVF) según cilindrada y nº cilindros.
    """
    # Fórmula estándar aproximada usada en España
    return 0.08 * cylinders * ((engine_cc / cylinders) ** 0.6)


def ivtm_from_cvf(cvf: float, bracket_table: List[Dict[str, float]]) -> float:
    """
    Obtener IVTM (impuesto de circulación) a partir de CVF y tabla de tramos municipal.
    """
    for row in bracket_table:
        if cvf <= row['cvf_max']:
            return float(row['cost'])
    # fallback
    return float(bracket_table[-1]['cost'])


def itv_annualized(age_years: float, cost_4_10_biennial: float = 40.0, cost_over10_annual: float = 35.0) -> float:
    """
    Coste ITV anualizado según edad del coche (aproximación).
    """
    if age_years < 4:
        return 0.0
    elif age_years < 10:
        return cost_4_10_biennial / 2.0
    else:
        return cost_over10_annual


def preventive_cost_per_km_from_odometer(odometer_km: float, brackets: List[Dict[str, float]]) -> float:
    """
    Determinar el coste preventivo por km según tramos de odómetro.
    """
    for row in brackets:
        if odometer_km <= row['km_max']:
            return float(row['cost_per_km'])
    return float(brackets[-1]['cost_per_km'])


def maintenance_preventive(annual_km: float, cost_per_km: float) -> float:
    """Coste anual de mantenimiento preventivo."""
    return annual_km * cost_per_km


def maintenance_corrective(age_years: float,
                           odometer_km: float,
                           annual_km: float,
                           corr_base: float,
                           alpha_age: float,
                           alpha_odo: float,
                           threshold_events: Optional[List[Dict]] = None,
                           default_prorate_years: int = 3) -> float:
    """
    Coste anual de mantenimiento correctivo, incluyendo eventos por umbral de km.
    """
    # Componente base multiplicativo por edad y desgaste acumulado
    ln_term = math.log(1 + odometer_km / 10000.0)
    base_component = corr_base * (1.0 + alpha_age * age_years + alpha_odo * ln_term)
    
    # Provisiones por piezas umbral
    provisions = 0.0
    if threshold_events:
        for ev in threshold_events:
            thr = float(ev.get('threshold_km', 0))
            cost = float(ev.get('cost', 0))
            prorate = int(ev.get('prorate_years', default_prorate_years))
            
            # Si el umbral cae este año
            if odometer_km < thr <= (odometer_km + annual_km):
                provisions += cost
            # Si el umbral ya fue superado, prorratear mientras no sepamos si ya se hizo
            elif odometer_km >= thr:
                provisions += cost / max(prorate, 1)
    
    return base_component + provisions


def tyres_cost(annual_km: float, tyre_set_cost: float, tyre_life_km: float) -> float:
    """Coste anual por desgaste de neumáticos."""
    if tyre_life_km <= 0:
        return 0.0
    return tyre_set_cost * (annual_km / tyre_life_km)


def insurance_cost(ins_base: float, age_years: float, power_kw: float, beta_age: float, beta_power: float) -> float:
    """Coste anual de seguro con coeficientes paramétricos."""
    return ins_base * (1.0 + beta_age * age_years + beta_power * (power_kw / 100.0))


def taxes_cost(engine_cc: float,
               cylinders: int,
               age_years: float,
               ivtm_brackets: List[Dict[str, float]],
               itv_cost_4_10_biennial: float = 40.0,
               itv_cost_over10_annual: float = 35.0) -> float:
    """Coste anual de impuestos (IVTM + ITV anualizada)."""
    cvf = compute_cvf(engine_cc, cylinders)
    ivtm = ivtm_from_cvf(cvf, ivtm_brackets)
    itv = itv_annualized(age_years, itv_cost_4_10_biennial, itv_cost_over10_annual)
    return ivtm + itv


# ===============================
# Simulaciones: compra y renting
# ===============================

@dataclass
class VehicleSpec:
    """Especificación del vehículo."""
    engine_cc: float
    cylinders: int
    power_kw: float
    fuel_type: Literal['gasoline', 'diesel', 'hybrid', 'electric'] = 'gasoline'


@dataclass
class MaintenanceSpec:
    """Parámetros de mantenimiento y neumáticos."""
    prev_brackets: List[Dict[str, float]] = field(default_factory=lambda: [
        {'km_max': 30000, 'cost_per_km': 0.030},
        {'km_max': 60000, 'cost_per_km': 0.035},
        {'km_max': 100000, 'cost_per_km': 0.045},
        {'km_max': 150000, 'cost_per_km': 0.055},
        {'km_max': float('inf'), 'cost_per_km': 0.065},
    ])
    corr_base: float = 200.0
    alpha_age: float = 0.10
    alpha_odo: float = 0.25
    threshold_events: List[Dict] = field(default_factory=lambda: [
        {'name': 'timing_belt', 'threshold_km': 100000, 'cost': 600, 'prorate_years': 3},
        {'name': 'shocks', 'threshold_km': 100000, 'cost': 700, 'prorate_years': 3},
    ])
    tyre_set_cost: float = 350.0
    tyre_life_km: float = 40000.0


@dataclass
class InsuranceSpec:
    """Parámetros del seguro."""
    ins_base: float = 500
    beta_age: float = 0.02
    beta_power: float = 0.05


@dataclass
class TaxesSpec:
    """Parámetros de impuestos."""
    ivtm_brackets: List[Dict[str, float]] = field(default_factory=lambda: [
        {'cvf_max': 8, 'cost': 30},
        {'cvf_max': 12, 'cost': 65},
        {'cvf_max': 16, 'cost': 95},
        {'cvf_max': 20, 'cost': 125},
        {'cvf_max': float('inf'), 'cost': 160},
    ])
    itv_cost_4_10_biennial: float = 40.0
    itv_cost_over10_annual: float = 35.0


@dataclass
class UsageSpec:
    """Uso del vehículo."""
    age_years: float
    annual_km: float
    odometer_km: float


@dataclass
class PurchaseFinanceSpec:
    """
    Parámetros de financiación para compra.
    
    Modes
    -----
    - 'amortized': usa price, down_payment, loan_rate_annual, loan_term_months.
    - 'custom': usa down_payment, installment_amount, installment_count, balloon_final.
      Se asume que los importes introducidos ya incorporan cualquier interés/comisión.
    """
    # Modo
    mode: Literal['amortized', 'custom'] = 'amortized'
    
    # Comunes
    price: float = 0.0
    down_payment: float = 0.0
    
    # Amortizado (francés)
    loan_rate_annual: float = 0.0
    loan_term_months: int = 0
    
    # Personalizado
    installment_amount: float = 0.0
    installment_count: int = 0
    balloon_final: float = 0.0


@dataclass
class DepreciationSpec:
    """Parámetros de depreciación."""
    method: Literal['declining', 'straight'] = 'declining'
    # Para 'declining'
    declining_rate: float = 0.10
    floor_fraction: float = 0.26
    # Para 'straight'
    life_years: float = 5
    salvage_fraction: float = 0.26


@dataclass
class RentingSpec:
    """Parámetros de renting."""
    monthly_fee: float
    months: int
    upfront_fee: float = 0.0
    includes_insurance: bool = True
    includes_maintenance: bool = True
    includes_tyres: bool = True
    includes_taxes: bool = True
    annual_km_allowance: float = 10000.0
    excess_km_cost: float = 0.05  # €/km excedente


def one_year_operating_cost(vehicle: VehicleSpec,
                            maint: MaintenanceSpec,
                            ins: InsuranceSpec,
                            taxes: TaxesSpec,
                            usage: UsageSpec) -> Dict[str, float]:
    """
    Costes operativos desglosados para un año dado.
    """
    # Coste preventivo dependiendo del odómetro
    c_prev_km = preventive_cost_per_km_from_odometer(usage.odometer_km, maint.prev_brackets)
    c_prev = maintenance_preventive(usage.annual_km, c_prev_km)
    
    # Coste correctivo con provisiones por umbrales
    c_corr = maintenance_corrective(
        age_years=usage.age_years,
        odometer_km=usage.odometer_km,
        annual_km=usage.annual_km,
        corr_base=maint.corr_base,
        alpha_age=maint.alpha_age,
        alpha_odo=maint.alpha_odo,
        threshold_events=maint.threshold_events
    )
    
    # Neumáticos
    c_tyres = tyres_cost(usage.annual_km, maint.tyre_set_cost, maint.tyre_life_km)
    
    # Seguro
    c_ins = insurance_cost(ins.ins_base, usage.age_years, vehicle.power_kw, ins.beta_age, ins.beta_power)
    
    # Impuestos
    c_tax = taxes_cost(vehicle.engine_cc, vehicle.cylinders, usage.age_years, taxes.ivtm_brackets,
                       taxes.itv_cost_4_10_biennial, taxes.itv_cost_over10_annual)
    
    total = c_prev + c_corr + c_tyres + c_ins + c_tax
    return dict(preventive=c_prev, corrective=c_corr, tyres=c_tyres, insurance=c_ins, taxes=c_tax, total=total)


def simulate_purchase_over_horizon(years: int,
                                   vehicle: VehicleSpec,
                                   maint: MaintenanceSpec,
                                   ins: InsuranceSpec,
                                   taxes: TaxesSpec,
                                   usage: UsageSpec,
                                   finance: PurchaseFinanceSpec,
                                   depreciation: DepreciationSpec) -> Dict[str, float]:
    """
    Simular compra durante un horizonte de años con costes operativos, pagos y valor residual.
    Incluye dos modos de financiación: 'amortized' o 'custom'.
    """
    # Copias para simular año a año
    age = float(usage.age_years)
    odo = float(usage.odometer_km)
    annual_km = float(usage.annual_km)
    
    # Costes operativos acumulados
    op_costs_sum = 0.0
    
    # Simulación anual
    for _ in range(years):
        year_usage = UsageSpec(age_years=age, annual_km=annual_km, odometer_km=odo)
        ycost = one_year_operating_cost(vehicle, maint, ins, taxes, year_usage)
        op_costs_sum += ycost['total']
        # Actualizar edad y odómetro
        age += 1.0
        odo += annual_km
    
    # Principal
    principal = max(finance.price - finance.down_payment, 0.0)
    months = years * 12
    
    if finance.mode == 'amortized':
        # Pagos de préstamo en el horizonte
        monthly_payment = annuity_payment(
            principal=principal,
            annual_rate=finance.loan_rate_annual,
            term_months=finance.loan_term_months
        )
        months_paid = min(months, max(finance.loan_term_months, 0))
        loan_payments = monthly_payment * months_paid
        loan_balance_end = loan_balance(principal,
                                        finance.loan_rate_annual,
                                        finance.loan_term_months,
                                        months_paid)
    else:
        # Modo personalizado: se asume que los importes ya incorporan intereses/comisiones
        monthly_payment = finance.installment_amount
        months_paid = min(months, max(finance.installment_count, 0))
        # Pagos mensuales dentro del horizonte
        loan_payments = monthly_payment * months_paid
        # Balloon si cae dentro del horizonte (al final de las cuotas pactadas)
        if months >= finance.installment_count:
            loan_payments += finance.balloon_final
            paid_principal = loan_payments
        else:
            # Si el horizonte termina antes de la cuota final
            paid_principal = loan_payments
        # Saldo restante simple (sin interés explícito: los importes ya lo incluyen)
        loan_balance_end = max(principal - paid_principal, 0.0)
    
    # Valor residual del coche al final del horizonte (edad final)
    age_end = usage.age_years + years
    if depreciation.method == 'declining':
        residual_value = declining_balance_value(finance.price, age_end, depreciation.declining_rate, depreciation.floor_fraction)
    else:
        residual_value = straight_line_value(finance.price, age_end, depreciation.life_years, depreciation.salvage_fraction)
    
    # "Equity" al final: valor menos saldo pendiente (si negativo, considerar 0 como equity)
    equity_end = max(residual_value - loan_balance_end, 0.0)
    
    # Salida de caja: entrada + pagos + costes operativos
    cash_out = finance.down_payment + loan_payments + op_costs_sum
    
    # Coste neto: salida de caja - equity
    net_cost = cash_out - equity_end
    
    return dict(
        op_costs=op_costs_sum,
        monthly_payment=monthly_payment,
        loan_payments=loan_payments,
        loan_balance_end=loan_balance_end,
        residual_value=residual_value,
        equity_end=equity_end,
        cash_out=cash_out,
        net_cost=net_cost
    )


def simulate_renting_over_horizon(years: int,
                                  renting: RentingSpec,
                                  vehicle: VehicleSpec,
                                  maint: MaintenanceSpec,
                                  ins: InsuranceSpec,
                                  taxes: TaxesSpec,
                                  usage: UsageSpec) -> Dict[str, float]:
    """Simular renting durante un horizonte de años."""
    months = years * 12
    months_effective = min(months, renting.months)
    fee_payments = renting.monthly_fee * months_effective
    upfront = renting.upfront_fee
    
    # Costes adicionales si el renting NO incluye ciertos servicios
    add_on_costs = 0.0
    
    yearly_usage = UsageSpec(age_years=usage.age_years, annual_km=usage.annual_km, odometer_km=usage.odometer_km)
    year_costs = one_year_operating_cost(vehicle, maint, ins, taxes, yearly_usage)
    
    if not renting.includes_maintenance:
        add_on_costs += (year_costs['preventive'] + year_costs['corrective']) * years
    if not renting.includes_tyres:
        add_on_costs += year_costs['tyres'] * years
    if not renting.includes_insurance:
        add_on_costs += year_costs['insurance'] * years
    if not renting.includes_taxes:
        add_on_costs += year_costs['taxes'] * years
    
    # Exceso de km (si aplica)
    excess_km_cost = 0.0
    if renting.annual_km_allowance is not None and renting.excess_km_cost is not None:
        for _ in range(years):
            excess = max(0.0, usage.annual_km - renting.annual_km_allowance)
            excess_km_cost += excess * renting.excess_km_cost
    
    total_cost = fee_payments + upfront + add_on_costs + excess_km_cost
    
    return dict(
        fee_payments=fee_payments,
        upfront_fee=upfront,
        add_on_costs=add_on_costs,
        excess_km_cost=excess_km_cost,
        total_cost=total_cost
    )


def compare_purchase_vs_renting(scenario_name: str,
                                years: int,
                                vehicle: VehicleSpec,
                                maint: MaintenanceSpec,
                                ins: InsuranceSpec,
                                taxes: TaxesSpec,
                                usage: UsageSpec,
                                finance: PurchaseFinanceSpec,
                                depreciation: DepreciationSpec,
                                renting: RentingSpec) -> Dict[str, float]:
    """Comparar compra vs renting para un escenario dado."""
    purchase_res = simulate_purchase_over_horizon(
        years, vehicle, maint, ins, taxes, usage, finance, depreciation
    )
    renting_res = simulate_renting_over_horizon(
        years, renting, vehicle, maint, ins, taxes, usage
    )
    
    delta = renting_res['total_cost'] - purchase_res['net_cost']
    
    return {
        'scenario': scenario_name,
        'horizon_years': years,
        # Compra
        'purchase_op_costs': round(purchase_res['op_costs'], 2),
        'purchase_loan_payments': round(purchase_res['loan_payments'], 2),
        'purchase_loan_balance_end': round(purchase_res['loan_balance_end'], 2),
        'purchase_residual_value': round(purchase_res['residual_value'], 2),
        'purchase_equity_end': round(purchase_res['equity_end'], 2),
        'purchase_cash_out': round(purchase_res['cash_out'], 2),
        'purchase_net_cost': round(purchase_res['net_cost'], 2),
        # Renting
        'renting_fee_payments': round(renting_res['fee_payments'], 2),
        'renting_upfront_fee': round(renting_res['upfront_fee'], 2),
        'renting_add_on_costs': round(renting_res['add_on_costs'], 2),
        'renting_excess_km_cost': round(renting_res['excess_km_cost'], 2),
        'renting_total_cost': round(renting_res['total_cost'], 2),
        # Comparación
        'delta_renting_minus_purchase': round(delta, 2)
    }
