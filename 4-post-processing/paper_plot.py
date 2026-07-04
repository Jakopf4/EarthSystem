# Können auch verschiedene Files werden, aber alle plot/video Funktionen
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors

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


def plot_combined_precipitation_loss_from_csv(scenario_1: str = "Raw", scenario_2: str = "BaU", metric_threshold: float = 1.1514) -> None:
    """
    Analysiert den >25% Niederschlagsverlust (Time of Emergence) 
    direkt aus den vorberechneten CSV-Dateien (Pandas).
    """
    scenarios = [245, 370, 585]
    years_future = np.arange(2030, 2100) # Passe das an dein gewünschtes Endjahr an
    
    # Pfade an deine Ordnerstruktur anpassen
    hist_file = "../3-processing/processed-data/timeseries_historical/historical_spatial_average_per_cell.csv"
    data_dir = "../3-processing/processed-data/"
    out_dir = "../4-post-processing/Plots/"
    
    print("\n--- SCHRITT 1: Historische Basislinie aus CSV laden ---")
    if not os.path.exists(hist_file):
        print(f"FEHLER: Historische CSV nicht gefunden unter: {hist_file}")
        return
        
    df_hist = pd.read_csv(hist_file)
    # Koordinaten runden für sauberen Merge
    df_hist['lat_round'] = df_hist['lat'].round(4)
    df_hist['lon_round'] = df_hist['lon'].round(4)
    
    # Wir brauchen nur die Koordinaten und den Threshold
    df_hist['threshold_prec'] = df_hist['mean_annual_prec'] * 0.75 
    
    print("\n--- SCHRITT 2: Zukunftsdaten (CSVs) vergleichen ---")
    
    pct_base = {ssp: [] for ssp in scenarios}
    pct_sec = {ssp: [] for ssp in scenarios}
    
    num_cells = len(df_hist)
    toe_base = {ssp: np.full(num_cells, 9999) for ssp in scenarios}
    toe_sec = {ssp: np.full(num_cells, 9999) for ssp in scenarios}
    
    # Wir speichern uns die exakten lats/lons für den Plot in der gleichen Reihenfolge wie in df_hist
    plot_lats = df_hist['lat'].values
    plot_lons = df_hist['lon'].values

    for ssp in scenarios:
        print(f"  -> Berechne SSP{ssp} ({scenario_1} vs {scenario_2})...")
        
        for y in years_future:
            f_base = os.path.join(data_dir, f"metrics_{scenario_1}", f"network_metrics_ssp{ssp}_decade{y}_{scenario_1}.csv")
            f_sec = os.path.join(data_dir, f"metrics_{scenario_2}", f"network_metrics_ssp{ssp}_decade{y}_{scenario_2}.csv")
            
            if os.path.exists(f_base) and os.path.exists(f_sec):
                # Zukunftsdaten laden
                df_b = pd.read_csv(f_base)
                df_s = pd.read_csv(f_sec)
                
                # Wir filtern nach einem Threshold, damit wir jede Zelle nur 1x pro Jahr haben
                # (Die 'prec' Spalte ist für alle Thresholds derselben Zelle identisch)
                df_b = df_b[np.isclose(df_b['threshold'], metric_threshold, atol=0.01)].copy()
                df_s = df_s[np.isclose(df_s['threshold'], metric_threshold, atol=0.01)].copy()
                
                df_b['lat_round'] = df_b['lat'].round(4)
                df_b['lon_round'] = df_b['lon'].round(4)
                df_s['lat_round'] = df_s['lat'].round(4)
                df_s['lon_round'] = df_s['lon'].round(4)
                
                # Merge mit der historischen Tabelle, um die Reihenfolge perfekt abzugleichen
                merged_b = pd.merge(df_hist, df_b, on=['lat_round', 'lon_round'], how='left')
                merged_s = pd.merge(df_hist, df_s, on=['lat_round', 'lon_round'], how='left')
                
                p_base = merged_b['prec'].values*12
                p_sec = merged_s['prec'].values*12
                t_prec = merged_b['threshold_prec'].values
                
                # 1. Schwellenwert prüfen (Ignoriere NaNs)
                with np.errstate(invalid='ignore'):
                    is_crit_base = p_base < t_prec
                    is_crit_sec = p_sec < t_prec

                if ssp == 245 and y == 2030:
                    print(f"\n--- BEISPIEL-ZELLE 0 IN 2030 (SSP245) ---")
                    print(f"Historischer Schnitt: {df_hist['mean_annual_prec'].iloc[0]:.0f} mm")
                    print(f"Kritischer Schwellenwert (75%): {t_prec[0]:.0f} mm")
                    print(f"Niederschlag Raw-Szenario: {p_base[0]:.0f} mm")
                    print(f"Niederschlag Sec-Szenario: {p_sec[0]:.0f} mm")
                    print(f"Verlust bei Raw > 25%? {is_crit_base[0]}")
                    print(f"Verlust bei Sec > 25%? {is_crit_sec[0]}\n")
                
                # 2. Flächenanteil speichern
                pct_base[ssp].append((np.sum(is_crit_base) / num_cells) * 100)
                pct_sec[ssp].append((np.sum(is_crit_sec) / num_cells) * 100)
                
                # 3. Time of Emergence aktualisieren
                mask_base = is_crit_base & (toe_base[ssp] == 9999)
                toe_base[ssp][mask_base] = y
                
                mask_sec = is_crit_sec & (toe_sec[ssp] == 9999)
                toe_sec[ssp][mask_sec] = y
            else:
                pct_base[ssp].append(np.nan)
                pct_sec[ssp].append(np.nan)

    print("\n--- SCHRITT 3: Erstelle Visualisierung ---")
    fig = plt.figure(figsize=(16, 16), dpi=150)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 1, 1], hspace=0.35)
    
    # ----------------------------------------------------
    # PLOT OBEN: Zeitreihe
    # ----------------------------------------------------
    ax_ts = fig.add_subplot(gs[0, :])
    colors = {245: "#f5a623", 370: "#d0021b", 585: "#9013fe"}
    
    for ssp in scenarios:
        valid_idx = ~np.isnan(pct_base[ssp])
        valid_years = years_future[valid_idx]
        
        if len(valid_years) > 0:
            ax_ts.plot(valid_years, np.array(pct_base[ssp])[valid_idx], color=colors[ssp], 
                       linewidth=2.5, linestyle="-", label=f"SSP{ssp} ({scenario_1})")
            
            ax_ts.plot(valid_years, np.array(pct_sec[ssp])[valid_idx], color=colors[ssp], 
                       linewidth=3.0, linestyle=":", label=f"SSP{ssp} ({scenario_2})")
        
    ax_ts.set_title(f"Amazon Area Experiencing >25% Precipitation Loss\n({scenario_1} vs {scenario_2})", fontsize=16, fontweight="bold", pad=10)
    ax_ts.set_ylabel("Affected Area [% of Grid Cells]", fontsize=12)
    ax_ts.set_xlabel("Year", fontsize=12)
    ax_ts.grid(True, linestyle="--", alpha=0.5)
    ax_ts.spines["top"].set_visible(False)
    ax_ts.spines["right"].set_visible(False)
    
    ax_ts.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), frameon=True, ncol=2, fontsize=11)
    ax_ts.set_xlim(years_future[0], years_future[-1])
    ax_ts.set_ylim(0, 100)
    
    # ----------------------------------------------------
    # PLOTS UNTEN: Time of Emergence Karten
    # ----------------------------------------------------
    cmap = plt.cm.YlOrRd 
    cmap.set_over("lightgrey")
    norm = mcolors.Normalize(vmin=years_future[0], vmax=years_future[-1])

    for idx, ssp in enumerate(scenarios):
        # REIHE 2: Scenario 1
        ax_map_base = fig.add_subplot(gs[1, idx])
        ax_map_base.scatter(
            x=plot_lons, y=plot_lats, c=toe_base[ssp], 
            cmap=cmap, norm=norm, s=100, marker="s", alpha=0.9, edgecolors="none"
        )
        ax_map_base.set_title(f"SSP {ssp} ({scenario_1})", fontsize=13, fontweight="bold")
        if idx == 0: ax_map_base.set_ylabel("Latitude", fontsize=10)
        ax_map_base.grid(True, linestyle="--", alpha=0.3)
        ax_map_base.set_xticklabels([]) 

        # REIHE 3: Scenario 2
        ax_map_sec = fig.add_subplot(gs[2, idx])
        sc = ax_map_sec.scatter(
            x=plot_lons, y=plot_lats, c=toe_sec[ssp], 
            cmap=cmap, norm=norm, s=100, marker="s", alpha=0.9, edgecolors="none"
        )
        ax_map_sec.set_title(f"SSP {ssp} ({scenario_2})", fontsize=13, fontweight="bold")
        if idx == 0: ax_map_sec.set_ylabel("Latitude", fontsize=10)
        ax_map_sec.set_xlabel("Longitude", fontsize=10)
        ax_map_sec.grid(True, linestyle="--", alpha=0.3)

    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.45])
    cbar = fig.colorbar(sc, cax=cbar_ax, extend='max')
    cbar.set_label("Year of Crossing -25% Threshold\n(Grey = Never)", fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"Precipitation_Loss_Analysis_{scenario_1}_vs_{scenario_2}.png")
    
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot erfolgreich gespeichert: {out_path}")
    



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

# Plot Robustness Analysis (Precipitation Loss)
    plot_combined_precipitation_loss_from_csv(scenario_1="Raw", scenario_2="Sec", metric_threshold=1.1514)
    plot_combined_precipitation_loss_from_csv(scenario_1="Raw", scenario_2="BaU", metric_threshold=1.1514)
    plot_combined_precipitation_loss_from_csv(scenario_1="Raw", scenario_2="Gov", metric_threshold=1.1514)

    