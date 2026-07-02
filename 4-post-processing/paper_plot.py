# Können auch verschiedene Files werden, aber alle plot/video Funktionen
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_MAP_from_csv(threshold_val: float = 1.14, scenario_1: str = "Raw", scenario_2: str = "Sec") -> None:
    """
    Erstellt den Time-Series Plot für Mean Annual Precipitation
    direkt aus den vorprozessierten CSV-Dateien und speichert ihn als PNG.
    Erlaubt den dynamischen Vergleich zweier beliebiger Szenarien (z.B. "Raw" vs "Sec").
    """
    scenarios = [245, 370, 585]
    
    # Basis-Pfad zu deinen neuen processed-data
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    
    # Farben exakt wie in deiner Vorlage
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    # Lesbare Labels für die Legende
    scenario_labels = {
        "Raw": "No Deforestation (Raw)",
        "Sec": "With Deforestation (Sec)",
        "BaU": "Business as Usual (BaU)",
        "Gov": "Governance (Gov)"
    }
    
    fig, ax = plt.subplots(figsize=(11, 7), dpi=150)
    print(f"Erstelle Plot für Threshold {threshold_val} aus CSV-Daten ({scenario_1} vs {scenario_2})...")

    for ssp in scenarios:
        # Pfade zu den neuen CSV-Dateien dynamisch anhand der Argumente zusammenbauen
        file_1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
        file_2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
        
        if os.path.exists(file_1) and os.path.exists(file_2):
            df_1 = pd.read_csv(file_1)
            df_2 = pd.read_csv(file_2)
            
            df_1 = df_1[df_1['threshold'] == threshold_val].sort_values(by='year')
            df_2 = df_2[df_2['threshold'] == threshold_val].sort_values(by='year')
            
            # Plotten (x = year, y = prec * 12)
            label_1 = f"{scenario_labels.get(scenario_1, scenario_1)} - SSP{ssp}"
            label_2 = f"{scenario_labels.get(scenario_2, scenario_2)} - SSP{ssp}"
            
            ax.plot(df_1['year'], df_1['prec'] * 12, color=colors[ssp], 
                    linestyle="-", linewidth=2, label=label_1)
            
            ax.plot(df_2['year'], df_2['prec'] * 12, color=colors[ssp], 
                    linestyle=":", linewidth=2.5, label=label_2)
        else:
            print(f"Warnung: Dateien für SSP{ssp} nicht gefunden unter:\n{file_1}\n{file_2}")

    # =====================================================================
    # KOSMETIK & SPEICHERN
    # =====================================================================
    ax.axhline(y=1800, color="#b8b28f", linestyle="--", linewidth=2.5, label="Bistable Threshold (1800 mm)")
    
    ax.set_title(f"Mean Annual Precipitation Comparison ({scenario_1} vs {scenario_2})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Mean Annual Precipitation [mm]", fontsize=12)
    
    ax.set_xlim(2030, 2100)
    ax.set_ylim(1000, 2000)
    ax.set_xticks(range(2030, 2101, 5))
    
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legende nach UNTEN außerhalb des Plots verschieben
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"MAP_Comparison_{scenario_1}_vs_{scenario_2}_Threshold_{threshold_val}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot erfolgreich gespeichert unter: {save_path}")


def plot_FFL_comparison(threshold_val: float = 1.14, scenario_1: str = "Raw", scenario_2: str = "Sec", show_monthly: bool = True) -> None:
    """
    Plottet die Entwicklung der Feed-Forward Loops (FFLs) über die Zeit (ABSOLUTE WERTE).
    Über das Argument 'show_monthly' können die saisonalen Schwankungen ein-/ausgeschaltet werden.
    """
    scenarios = [245, 370, 585]
    
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    scenario_labels = {
        "Raw": "No Deforestation (Raw)",
        "Sec": "With Deforestation (Sec)",
        "BaU": "Business as Usual (BaU)",
        "Gov": "Governance (Gov)"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    mode_str = "MIT" if show_monthly else "OHNE"
    print(f"Erstelle absoluten FFL-Plot für Threshold {threshold_val} ({scenario_1} vs {scenario_2}, {mode_str} monatliche Daten)...")

    for ssp in scenarios:
        file_y1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
        file_y2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
        
        if os.path.exists(file_y1) and os.path.exists(file_y2):
            df_y1 = pd.read_csv(file_y1)
            df_y2 = pd.read_csv(file_y2)
            
            df_y1 = df_y1[df_y1['threshold'] == threshold_val].sort_values(by='year')
            df_y2 = df_y2[df_y2['threshold'] == threshold_val].sort_values(by='year')
            
            if df_y1.empty or df_y2.empty:
                continue

            # --- OPTIONAL: MONATLICHE DATEN PLOTTEN ---
            if show_monthly:
                file_m1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
                file_m2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
                
                if os.path.exists(file_m1) and os.path.exists(file_m2):
                    df_m1 = pd.read_csv(file_m1)
                    df_m2 = pd.read_csv(file_m2)
                    
                    df_m1 = df_m1[df_m1['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    df_m2 = df_m2[df_m2['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    
                    time_m1 = df_m1['year'] + (df_m1['month'] - 1) / 12.0
                    time_m2 = df_m2['year'] + (df_m2['month'] - 1) / 12.0
                    
                    ax.plot(time_m1, df_m1['ffl'], color=colors[ssp], alpha=0.15, linestyle="-")
                    ax.plot(time_m2, df_m2['ffl'], color=colors[ssp], alpha=0.15, linestyle=":")
            
            # --- PLOTTEN (Jahre - Vordergrund) ---
            label_1 = f"{scenario_labels.get(scenario_1, scenario_1)} - SSP{ssp}"
            label_2 = f"{scenario_labels.get(scenario_2, scenario_2)} - SSP{ssp}"
            
            ax.plot(df_y1['year'], df_y1['ffl'], color=colors[ssp], linestyle="-", linewidth=2.5, label=label_1)
            ax.plot(df_y2['year'], df_y2['ffl'], color=colors[ssp], linestyle=":", linewidth=3.0, label=label_2)
            
        else:
            print(f"Warnung: Fehlende Dateien für SSP{ssp}. Überspringe...")

    # =====================================================================
    ax.set_title(f"Global Feed-Forward Loops (FFL) Over Time\n({scenario_1} vs {scenario_2})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average FFL Count per Node", fontsize=12)
    ax.set_xlim(2030, 2100)
    ax.set_xticks(range(2030, 2101, 5))
    
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legende nach UNTEN außerhalb des Plots verschieben
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_with_monthly" if show_monthly else "_yearly_only"
    save_path = os.path.join(out_dir, f"FFL_Absolute_{scenario_1}_vs_{scenario_2}_Threshold_{threshold_val}{suffix}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot erfolgreich gespeichert unter: {save_path}")


def plot_FFL_relative_comparison(threshold_val: float = 1.14, scenario_1: str = "Raw", scenario_2: str = "Sec", show_monthly: bool = True) -> None:
    """
    Plottet die Entwicklung der Feed-Forward Loops (FFLs) über die Zeit (RELATIVE WERTE %).
    Zeigt die Werte prozentual im Vergleich zum Startjahr (100%).
    Über das Argument 'show_monthly' können die saisonalen Schwankungen ein-/ausgeschaltet werden.
    """
    scenarios = [245, 370, 585]
    
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    scenario_labels = {
        "Raw": "No Deforestation (Raw)",
        "Sec": "With Deforestation (Sec)",
        "BaU": "Business as Usual (BaU)",
        "Gov": "Governance (Gov)"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    mode_str = "MIT" if show_monthly else "OHNE"
    print(f"Erstelle relativen FFL-Plot für Threshold {threshold_val} ({scenario_1} vs {scenario_2}, {mode_str} monatliche Daten)...")

    for ssp in scenarios:
        # Pfade zusammenbauen
        file_y1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
        file_y2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
        
        if os.path.exists(file_y1) and os.path.exists(file_y2):
            
            # Einlesen und Filtern
            df_y1 = pd.read_csv(file_y1)
            df_y2 = pd.read_csv(file_y2)
            
            df_y1 = df_y1[df_y1['threshold'] == threshold_val].sort_values(by='year')
            df_y2 = df_y2[df_y2['threshold'] == threshold_val].sort_values(by='year')
            
            if df_y1.empty or df_y2.empty:
                continue
            
            # --- BASELINE ERMITTELN (Wert des ersten Jahres von Scenario 1) ---
            base_val_1 = df_y1.iloc[0]['ffl']
            base_val_2 = df_y1.iloc[0]['ffl'] 
            
            # --- PROZENTUALE WERTE BERECHNEN (JAHRE) ---
            y1_pct = (df_y1['ffl'] / base_val_1) * 100
            y2_pct = (df_y2['ffl'] / base_val_2) * 100
            
            # --- OPTIONAL: MONATLICHE DATEN PLOTTEN ---
            if show_monthly:
                file_m1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
                file_m2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
                
                if os.path.exists(file_m1) and os.path.exists(file_m2):
                    df_m1 = pd.read_csv(file_m1)
                    df_m2 = pd.read_csv(file_m2)
                    
                    df_m1 = df_m1[df_m1['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    df_m2 = df_m2[df_m2['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    
                    m1_pct = (df_m1['ffl'] / base_val_1) * 100
                    m2_pct = (df_m2['ffl'] / base_val_2) * 100
                    
                    time_m1 = df_m1['year'] + (df_m1['month'] - 1) / 12.0
                    time_m2 = df_m2['year'] + (df_m2['month'] - 1) / 12.0
                    
                    ax.plot(time_m1, m1_pct, color=colors[ssp], alpha=0.15, linestyle="-")
                    ax.plot(time_m2, m2_pct, color=colors[ssp], alpha=0.15, linestyle=":")
            
            # --- PLOTTEN (Jahre - immer im Vordergrund) ---
            label_1 = f"{scenario_labels.get(scenario_1, scenario_1)} - SSP{ssp}"
            label_2 = f"{scenario_labels.get(scenario_2, scenario_2)} - SSP{ssp}"
            
            ax.plot(df_y1['year'], y1_pct, color=colors[ssp], linestyle="-", linewidth=2.5, label=label_1)
            ax.plot(df_y2['year'], y2_pct, color=colors[ssp], linestyle=":", linewidth=3.0, label=label_2)
            
        else:
            print(f"Warnung: Jährliche Dateien für SSP{ssp} nicht gefunden. Überspringe...")

    # =====================================================================
    # KOSMETIK & SPEICHERN
    # =====================================================================
    ax.axhline(y=100, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Initial Value (100%)")
    
    ax.set_title(f"Relative Change of Feed-Forward Loops (FFL)\n({scenario_1} vs {scenario_2})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("FFL Count [% of Initial Year]", fontsize=12)
    
    ax.set_xlim(2030, 2100)
    ax.set_xticks(range(2030, 2101, 5))
    
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legende nach UNTEN außerhalb des Plots verschieben
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_with_monthly" if show_monthly else "_yearly_only"
    save_path = os.path.join(out_dir, f"FFL_Relative_{scenario_1}_vs_{scenario_2}_Threshold_{threshold_val}{suffix}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot erfolgreich gespeichert unter: {save_path}\n")


def plot_Clustering_comparison(threshold_val: float = 1.14, scenario_1: str = "Raw", scenario_2: str = "Sec", show_monthly: bool = True) -> None:
    """
    Plottet die Entwicklung des Clustering Coefficients über die Zeit (ABSOLUTE WERTE).
    Über das Argument 'show_monthly' können die saisonalen Schwankungen ein-/ausgeschaltet werden.
    """
    scenarios = [245, 370, 585]
    
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    scenario_labels = {
        "Raw": "No Deforestation (Raw)",
        "Sec": "With Deforestation (Sec)",
        "BaU": "Business as Usual (BaU)",
        "Gov": "Governance (Gov)"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    mode_str = "MIT" if show_monthly else "OHNE"
    print(f"Erstelle absoluten Clustering-Plot für Threshold {threshold_val} ({scenario_1} vs {scenario_2}, {mode_str} monatliche Daten)...")

    for ssp in scenarios:
        file_y1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
        file_y2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
        
        if os.path.exists(file_y1) and os.path.exists(file_y2):
            df_y1 = pd.read_csv(file_y1)
            df_y2 = pd.read_csv(file_y2)
            
            df_y1 = df_y1[df_y1['threshold'] == threshold_val].sort_values(by='year')
            df_y2 = df_y2[df_y2['threshold'] == threshold_val].sort_values(by='year')
            
            if df_y1.empty or df_y2.empty:
                continue

            # --- OPTIONAL: MONATLICHE DATEN PLOTTEN ---
            if show_monthly:
                file_m1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
                file_m2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
                
                if os.path.exists(file_m1) and os.path.exists(file_m2):
                    df_m1 = pd.read_csv(file_m1)
                    df_m2 = pd.read_csv(file_m2)
                    
                    df_m1 = df_m1[df_m1['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    df_m2 = df_m2[df_m2['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    
                    time_m1 = df_m1['year'] + (df_m1['month'] - 1) / 12.0
                    time_m2 = df_m2['year'] + (df_m2['month'] - 1) / 12.0
                    
                    ax.plot(time_m1, df_m1['clustering'], color=colors[ssp], alpha=0.15, linestyle="-")
                    ax.plot(time_m2, df_m2['clustering'], color=colors[ssp], alpha=0.15, linestyle=":")
            
            # --- PLOTTEN (Jahre - Vordergrund) ---
            label_1 = f"{scenario_labels.get(scenario_1, scenario_1)} - SSP{ssp}"
            label_2 = f"{scenario_labels.get(scenario_2, scenario_2)} - SSP{ssp}"
            
            ax.plot(df_y1['year'], df_y1['clustering'], color=colors[ssp], linestyle="-", linewidth=2.5, label=label_1)
            ax.plot(df_y2['year'], df_y2['clustering'], color=colors[ssp], linestyle=":", linewidth=3.0, label=label_2)
            
        else:
            print(f"Warnung: Fehlende Dateien für SSP{ssp}. Überspringe...")

    # =====================================================================
    ax.set_title(f"Global Clustering Coefficient Over Time\n({scenario_1} vs {scenario_2})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Average Clustering Coefficient", fontsize=12)
    ax.set_xlim(2030, 2100)
    ax.set_xticks(range(2030, 2101, 5))
    
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legende nach UNTEN außerhalb des Plots verschieben
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_with_monthly" if show_monthly else "_yearly_only"
    save_path = os.path.join(out_dir, f"Clustering_Absolute_{scenario_1}_vs_{scenario_2}_Threshold_{threshold_val}{suffix}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot erfolgreich gespeichert unter: {save_path}")


def plot_Clustering_relative_comparison(threshold_val: float = 1.14, scenario_1: str = "Raw", scenario_2: str = "Sec", show_monthly: bool = True) -> None:
    """
    Plottet die Entwicklung der Clustering Coefficients über die Zeit (RELATIVE WERTE %).
    Zeigt die Werte prozentual im Vergleich zum Startjahr (100%).
    Über das Argument 'show_monthly' können die saisonalen Schwankungen ein-/ausgeschaltet werden.
    """
    scenarios = [245, 370, 585]
    
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    scenario_labels = {
        "Raw": "No Deforestation (Raw)",
        "Sec": "With Deforestation (Sec)",
        "BaU": "Business as Usual (BaU)",
        "Gov": "Governance (Gov)"
    }
    
    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    mode_str = "MIT" if show_monthly else "OHNE"
    print(f"Erstelle relativen Clustering-Plot für Threshold {threshold_val} ({scenario_1} vs {scenario_2}, {mode_str} monatliche Daten)...")

    for ssp in scenarios:
        # Pfade zusammenbauen
        file_y1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
        file_y2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
        
        if os.path.exists(file_y1) and os.path.exists(file_y2):
            
            # Einlesen und Filtern
            df_y1 = pd.read_csv(file_y1)
            df_y2 = pd.read_csv(file_y2)
            
            df_y1 = df_y1[df_y1['threshold'] == threshold_val].sort_values(by='year')
            df_y2 = df_y2[df_y2['threshold'] == threshold_val].sort_values(by='year')
            
            if df_y1.empty or df_y2.empty:
                continue
            
            # --- BASELINE ERMITTELN (Wert des ersten Jahres von Scenario 1) ---
            base_val_1 = df_y1.iloc[0]['clustering']
            base_val_2 = df_y1.iloc[0]['clustering'] 
            
            # --- PROZENTUALE WERTE BERECHNEN (JAHRE) ---
            y1_pct = (df_y1['clustering'] / base_val_1) * 100
            y2_pct = (df_y2['clustering'] / base_val_2) * 100
            
            # --- OPTIONAL: MONATLICHE DATEN PLOTTEN ---
            if show_monthly:
                file_m1 = os.path.join(data_dir, f"timeseries_{scenario_1}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_1}.csv")
                file_m2 = os.path.join(data_dir, f"timeseries_{scenario_2}", f"global_monthly_metrics_timeseries_ssp{ssp}_{scenario_2}.csv")
                
                if os.path.exists(file_m1) and os.path.exists(file_m2):
                    df_m1 = pd.read_csv(file_m1)
                    df_m2 = pd.read_csv(file_m2)
                    
                    df_m1 = df_m1[df_m1['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    df_m2 = df_m2[df_m2['threshold'] == threshold_val].sort_values(by=['year', 'month'])
                    
                    m1_pct = (df_m1['clustering'] / base_val_1) * 100
                    m2_pct = (df_m2['clustering'] / base_val_2) * 100
                    
                    time_m1 = df_m1['year'] + (df_m1['month'] - 1) / 12.0
                    time_m2 = df_m2['year'] + (df_m2['month'] - 1) / 12.0
                    
                    ax.plot(time_m1, m1_pct, color=colors[ssp], alpha=0.15, linestyle="-")
                    ax.plot(time_m2, m2_pct, color=colors[ssp], alpha=0.15, linestyle=":")
            
            # --- PLOTTEN (Jahre - immer im Vordergrund) ---
            label_1 = f"{scenario_labels.get(scenario_1, scenario_1)} - SSP{ssp}"
            label_2 = f"{scenario_labels.get(scenario_2, scenario_2)} - SSP{ssp}"
            
            ax.plot(df_y1['year'], y1_pct, color=colors[ssp], linestyle="-", linewidth=2.5, label=label_1)
            ax.plot(df_y2['year'], y2_pct, color=colors[ssp], linestyle=":", linewidth=3.0, label=label_2)
            
        else:
            print(f"Warnung: Jährliche Dateien für SSP{ssp} nicht gefunden. Überspringe...")

    # =====================================================================
    # KOSMETIK & SPEICHERN
    # =====================================================================
    ax.axhline(y=100, color="black", linestyle="--", linewidth=1.5, alpha=0.7, label="Initial Value (100%)")
    
    ax.set_title(f"Relative Change of Clustering Coefficient\n({scenario_1} vs {scenario_2})", fontsize=15, fontweight="bold", pad=20)
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Clustering Coefficient [% of Initial Year]", fontsize=12)
    
    ax.set_xlim(2030, 2100)
    ax.set_xticks(range(2030, 2101, 5))
    
    ax.grid(True, linestyle="--", alpha=0.5, color="lightgray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Legende nach UNTEN außerhalb des Plots verschieben
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=True, fontsize=10, ncol=2)
    
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    suffix = "_with_monthly" if show_monthly else "_yearly_only"
    save_path = os.path.join(out_dir, f"Clustering_Relative_{scenario_1}_vs_{scenario_2}_Threshold_{threshold_val}{suffix}.png")
    
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot erfolgreich gespeichert unter: {save_path}\n")



if __name__ == "__main__":

# Plot MAP
    #plot_MAP_from_csv(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec")
    #plot_MAP_from_csv(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU")
    #plot_MAP_from_csv(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov")

# Plot absolute FFL 
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=False)
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=False) 
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=False)
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=True)
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=True)
    #plot_FFL_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=True)  

# Plot relative FFL
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=False)
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=False)
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=False)
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=True)
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=True)
    #plot_FFL_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=True)

# Plot absolute Clustering
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=False)
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=False)
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=False)
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=True)
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=True)
    #plot_Clustering_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=True)

# Plot relative Clustering
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=False)
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=False)
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=False)
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Sec", show_monthly=True)
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="BaU", show_monthly=True)
    #plot_Clustering_relative_comparison(threshold_val=1.14, scenario_1="Raw", scenario_2="Gov", show_monthly=True)