from vehicle_capex_opex import *


# ===============================
# COMPARACIÓN DE SEAT IBIZA
# ===============================

# Especificación de Seat Ibiza
vehicle = VehicleSpec(engine_cc=999, cylinders=3, power_kw=70, fuel_type='gasoline')
maint = MaintenanceSpec()
ins = InsuranceSpec(ins_base=1000.0, beta_age=0.02, beta_power=0.05)
taxes = TaxesSpec()

# Uso
new = UsageSpec(age_years=0, annual_km=10000.0, odometer_km=0)
used = UsageSpec(age_years=4, annual_km=10000.0, odometer_km=100000)
    
# Compra nuevo
finance_new = PurchaseFinanceSpec(
    mode='custom',
    price=14000.0,
    down_payment=3725.0,
    installment_amount=90.0,        # monto de la cuota mensual
    installment_count=48,           # número de cuotas
    balloon_final=10528.38          # cuota final
)

finance_used = PurchaseFinanceSpec(
    mode='custom',
    price=11790.0,
    down_payment=0.0,
    installment_amount=184.0,        # monto de la cuota mensual
    installment_count=120,           # número de cuotas
    balloon_final=0.0                # cuota final
)

# Renting
renting_no_upfront = RentingSpec(monthly_fee=221, months=72, upfront_fee=0.0,
                                    includes_insurance=True, includes_maintenance=True, includes_tyres=True, includes_taxes=True,
                                    annual_km_allowance=10000.0, excess_km_cost=0.06)

years = 6

# Simulación compra personalizada
res_new = simulate_purchase_over_horizon(years=years,
                                            vehicle=vehicle,
                                            maint=maint,
                                            ins=ins,
                                            taxes=taxes,
                                            usage=new,
                                            finance=finance_new,
                                            depreciation=DepreciationSpec(method='declining', declining_rate=0.10, floor_fraction=0.50))

res_used = simulate_purchase_over_horizon(years=years,
                                            vehicle=vehicle,
                                            maint=maint,
                                            ins=ins,
                                            taxes=taxes,
                                            usage=used,
                                            finance=finance_used,
                                            depreciation=DepreciationSpec(method='declining', declining_rate=0.10, floor_fraction=0.50))

# Comparativa rápida contra renting (solo un ejemplo)
new_vs_reting = compare_purchase_vs_renting(f"SEAT Ibiza - Nuevo vs Renting - {years} años", years, vehicle, maint, ins, taxes,
                                            new, finance_new, DepreciationSpec(), renting_no_upfront)

used_vs_reting = compare_purchase_vs_renting(f"SEAT Ibiza - Usado vs Renting - {years} años", years, vehicle, maint, ins, taxes,
                                            used, finance_used, DepreciationSpec(), renting_no_upfront)

df = pd.DataFrame([
    {'scenario': 'Seat Ibiza - Financiado Nuevo', **res_new},
    {'scenario': 'Seat Ibiza - Financiado Usado', **res_used},
    new_vs_reting,
    used_vs_reting
])

df_comparations = pd.DataFrame([
    new_vs_reting,
    used_vs_reting
])

csv_path = "comparison_results.csv"
df.to_csv(csv_path, index=False)

# Imprimir comparaciones
columns = ["scenario", "purchase_op_costs", "purchase_net_cost", "renting_total_cost", "delta_renting_minus_purchase"]
print(df_comparations[columns])

