# Wie wenden wir deforestation auf die Daten an?

# Daten aus ../data/water und ../data/deforestation laden und in /pre-processing speichern (als .nc)

# 1. BaU (Worst Case Scenario)
# 2. Gov
# 3. Secondary Vegetation (Welches Scenario wurde hier genommen?)

import xarray as xr
import numpy as np
from pathlib import Path


def deforestation_network_BaU_one_step(scenario: int, year: int, month: int):
    """Berechnet den veränderten Wasserfluss und passt den Niederschlag (prec) 
    an den verringerten Wasser-Zufluss an.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        month (int): Monat des Szenarios.
    """
    nc_filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    txt_filepath = f"../data/deforestation/deforestation_all_BaU_{year}.txt"

    # Ursprüngliche Daten laden
    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # --- 1. Netzwerk-Matrix anpassen (One-Step) ---
    factor_matrix = 1 - deforest_values[:, np.newaxis]
    deforested_matrix = orig_matrix * factor_matrix

    # --- 2. Zufluss-Verhältnisse für den Niederschlag berechnen ---
    # Summe des ankommenden Wassers pro Zelle (vorher vs. nachher)
    orig_inflow = np.sum(orig_matrix, axis=0)
    new_inflow = np.sum(deforested_matrix, axis=0)

    # Verhältnisse (Ratios) berechnen
    ratios = np.zeros_like(orig_inflow)
    mask = orig_inflow != 0

    # Für alle Zellen mit vorherigem Zufluss: Ratio = Neu / Alt
    ratios[mask] = new_inflow[mask] / orig_inflow[mask]

    # WICHTIG: Für Zellen, die vorher schon gar keinen Zufluss aus dem Netzwerk hatten,
    # setzen wir den Ratio auf 1.0 (100%), damit ihr Niederschlag nicht fälschlicherweise auf 0 gesetzt wird.
    ratios[~mask] = 1.0

    # --- 3. NetCDF-Datei erstellen und speichern ---
    ds_new = ds.copy(deep=True)

    # Aktualisierte Netzwerk-Matrix einfügen
    ds_new["network"] = (ds["network"].dims, deforested_matrix)

    # Niederschlag ('prec') mit dem berechneten Zufluss-Verhältnis skalieren
    ds_new["prec"] = ds_new["prec"] * ratios

    # (Optional) Falls auch die 'evap' Variable lokal reduziert werden soll:
    ds_new["evap"] = ds_new["evap"] * (1 - deforest_values)

    output_filepath = f"./pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"

    ds_new.to_netcdf(output_filepath)

    ds.close()
    ds_new.close()

    print(f"Datei inkl. Niederschlagsanpassung erfolgreich erstellt: {output_filepath}")


def deforestation_network_Gov_one_step(scenario: int, year: int, month: int):
    """Berechnet den veränderten Wasserfluss und passt den Niederschlag (prec) 
    an den verringerten Wasser-Zufluss an.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        month (int): Monat des Szenarios.
    """
    nc_filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    txt_filepath = f"../data/deforestation/deforestation_all_Gov_{year}.txt"

    # Ursprüngliche Daten laden
    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # --- 1. Netzwerk-Matrix anpassen (One-Step) ---
    factor_matrix = 1 - deforest_values[:, np.newaxis]
    deforested_matrix = orig_matrix * factor_matrix

    # --- 2. Zufluss-Verhältnisse für den Niederschlag berechnen ---
    # Summe des ankommenden Wassers pro Zelle (vorher vs. nachher)
    orig_inflow = np.sum(orig_matrix, axis=0)
    new_inflow = np.sum(deforested_matrix, axis=0)

    # Verhältnisse (Ratios) berechnen
    ratios = np.zeros_like(orig_inflow)
    mask = orig_inflow != 0

    # Für alle Zellen mit vorherigem Zufluss: Ratio = Neu / Alt
    ratios[mask] = new_inflow[mask] / orig_inflow[mask]

    # WICHTIG: Für Zellen, die vorher schon gar keinen Zufluss aus dem Netzwerk hatten,
    # setzen wir den Ratio auf 1.0 (100%), damit ihr Niederschlag nicht fälschlicherweise auf 0 gesetzt wird.
    ratios[~mask] = 1.0

    # --- 3. NetCDF-Datei erstellen und speichern ---
    ds_new = ds.copy(deep=True)

    # Aktualisierte Netzwerk-Matrix einfügen
    ds_new["network"] = (ds["network"].dims, deforested_matrix)

    # Niederschlag ('prec') mit dem berechneten Zufluss-Verhältnis skalieren
    ds_new["prec"] = ds_new["prec"] * ratios

    # (Optional) Falls auch die 'evap' Variable lokal reduziert werden soll:
    ds_new["evap"] = ds_new["evap"] * (1 - deforest_values)

    output_filepath = f"./pre-processed-data/water_Gov/scenario_ssp{scenario}_decade{year}_month{month:02d}_Gov.nc"

    ds_new.to_netcdf(output_filepath)

    ds.close()
    ds_new.close()

    print(f"Datei inkl. Niederschlagsanpassung erfolgreich erstellt: {output_filepath}")


def deforestation_network_Sec(scenario: int, year: int, month: int):
    """Berechnet den veränderten Wasserfluss für das Sec-Szenario und passt 
    den Niederschlag (prec) und die Verdunstung (evap) basierend auf der 
    bereits aktualisierten Transportmatrix an.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        month (int): Monat des Szenarios.
    """
    nc_filepath = f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    sec_filepath = f"../data/water_secondary_vegetation/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"

    # 1. Originaldaten laden (Referenz für die ungestörten Werte)
    ds_orig = xr.open_dataset(nc_filepath)
    
    # 2. Updated Daten laden (Kopie erstellen, damit wir Werte überschreiben können)
    ds_sec = xr.open_dataset(sec_filepath).copy(deep=True)

    # Matrizen extrahieren
    orig_matrix = ds_orig["network"].values
    sec_matrix = ds_sec["network"].values

    # --- A. Niederschlag (prec) anpassen ---
    # Logik: Wie viel Wasser kommt flussabwärts bei jeder Zelle an? (Spaltensummen)
    orig_inflow = np.sum(orig_matrix, axis=0)
    new_inflow = np.sum(sec_matrix, axis=0)

    ratios_in = np.zeros_like(orig_inflow)
    mask_in = orig_inflow != 0
    
    # Ratio = Neu / Alt (Zufluss-Verhältnis)
    ratios_in[mask_in] = new_inflow[mask_in] / orig_inflow[mask_in]
    ratios_in[~mask_in] = 1.0  # Zellen ohne vorherigen Zufluss behalten 100% ihres Niederschlags

    # Niederschlag skalieren
    ds_sec["prec"] = ds_orig["prec"] * ratios_in

    # --- B. Verdunstung (evap) anpassen ---
    # Logik: Das Verhältnis des ausgehenden Wassers (Zeilensummen) entspricht exakt dem Faktor (1 - deforest_values)
    orig_outflow = np.sum(orig_matrix, axis=1)
    new_outflow = np.sum(sec_matrix, axis=1)

    ratios_out = np.zeros_like(orig_outflow)
    mask_out = orig_outflow != 0
    
    # Ratio = Neu / Alt (Lokaler Entwaldungs-Faktor)
    ratios_out[mask_out] = new_outflow[mask_out] / orig_outflow[mask_out]
    ratios_out[~mask_out] = 1.0  # Zellen ohne vorherigen Abfluss bleiben unverändert

    # Verdunstung skalieren
    ds_sec["evap"] = ds_orig["evap"] * ratios_out

    # --- C. Speichern ---
    # Ordner bei Bedarf erstellen
    output_dir = Path("./pre-processed-data/water_Sec")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filepath = output_dir / f"scenario_ssp{scenario}_decade{year}_month{month:02d}_Sec.nc"

    ds_sec.to_netcdf(output_filepath)

    # Dateien sauber schließen
    ds_orig.close()
    ds_sec.close()

    print(f"Datei inkl. Prec/Evap-Anpassung erfolgreich erstellt: {output_filepath}")


def deforestation_network_BaU_two_step(scenario: int, year: int, month: int):
    """Berechnet den veränderten Wasserfluss durch Entwaldung und speichert
    das Ergebnis in einer neuen NetCDF-Datei.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        month (int): Monat des Szenarios.
    """
    nc_filepath = (
        f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    )
    txt_filepath = f"../data/deforestation/deforestation_all_BaU_{year}.txt"

    # Ursprüngliche Daten laden
    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # Eingehendes Wasser ohne Entwaldung
    orig_col_sums = np.sum(orig_matrix, axis=0)

    # Kopie der Matrix für die Berechnungen
    current_matrix = orig_matrix.copy()

    # Transportwerte basierend auf Entwaldung anpassen
    factor_matrix = 1 - deforest_values[:, np.newaxis]
    current_matrix = current_matrix * factor_matrix

    # Eingehendes Wasser mit Entwaldung
    new_col_sums = np.sum(current_matrix, axis=0)

    ratios = np.zeros_like(orig_col_sums)

    # Division durch Null verhindern
    mask = orig_col_sums != 0
    ratios[mask] = new_col_sums[mask] / orig_col_sums[mask]

    # Wasserfluss basierend auf verringertem Zufluss reduzieren (Kaskade)
    deforested_matrix = current_matrix * ratios[:, np.newaxis]

    # --- AB HIER NEU: NetCDF-Datei erstellen und speichern ---

    # 1. Erstelle eine tiefe Kopie des originalen Datasets.
    # Dadurch bleiben alle Koordinaten, Dimensionen und globalen Metadaten erhalten.
    ds_new = ds.copy(deep=True)

    # 2. Ersetze die Werte der Variable "network" durch unsere neue Matrix.
    # Wir übergeben die Dimensionen der alten Variable, damit xarray die Struktur korrekt zuordnet.
    ds_new["network"] = (ds["network"].dims, deforested_matrix)

    # 3. Automatischen Dateinamen generieren, falls kein Pfad angegeben wurde
    output_filepath = f"./pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"

    # 4. Das modifizierte Dataset als neue NetCDF-Datei auf die Festplatte schreiben
    ds_new.to_netcdf(output_filepath)

    # 5. Dateien sauber schließen, um den Arbeitsspeicher freizugeben und File-Locks zu verhindern
    ds.close()
    ds_new.close()

    print(f"Datei erfolgreich erstellt und gespeichert unter: {output_filepath}")


def deforestation_network_BaU_converging(scenario: int, year: int, month: int):
    """Berechnet den veränderten Wasserfluss durch Entwaldung und speichert
    das Ergebnis in einer neuen NetCDF-Datei.

    Args:
        scenario (int): SSP Szenario-Nummer.
        year (int): Jahr des Szenarios.
        month (int): Monat des Szenarios.
    """
    nc_filepath = (
        f"../data/water/scenario_ssp{scenario}_decade{year}_month{month:02d}.nc"
    )
    txt_filepath = f"../data/deforestation/deforestation_all_BaU_{year}.txt"

    ds = xr.open_dataset(nc_filepath)
    data_deforestation = np.genfromtxt(txt_filepath)

    orig_matrix = ds["network"].values
    deforest_values = data_deforestation[:, 2]

    # Ursprünglichen Zufluss berechnen (Referenz für die Ratios)
    orig_col_sums = np.sum(orig_matrix, axis=0)

    # Maske für Division durch Null (wird nur einmal benötigt)
    mask = orig_col_sums != 0

    # 1. Basis-Matrix erstellen (Nur direkte, lokale Entwaldungseffekte)
    factor_matrix = 1 - deforest_values[:, np.newaxis]
    base_matrix = orig_matrix * factor_matrix

    # 2. Setup für die Iteration
    current_matrix = base_matrix.copy()
    tolerance = 1e-5  # Schwellenwert für die Konvergenz (anpassbar)
    max_diff = float('inf')
    iteration = 0
    max_iterations = 1000  # Sicherheitsnetz gegen Endlosschleifen

    # 3. Iterieren, bis die Änderungen kleiner als die Toleranz sind
    while max_diff > tolerance and iteration < max_iterations:
        prev_matrix = current_matrix.copy()

        # Aktuellen Zufluss im jetzigen Kaskaden-Schritt berechnen
        current_col_sums = np.sum(current_matrix, axis=0)

        # Ratio: Aktueller Zufluss / Ursprünglicher Zufluss
        ratios = np.zeros_like(orig_col_sums)
        ratios[mask] = current_col_sums[mask] / orig_col_sums[mask]

        # Neue Matrix berechnen: Die *Basis-Matrix* skaliert mit den neuen Zufluss-Ratios
        current_matrix = base_matrix * ratios[:, np.newaxis]

        # Konvergenz prüfen: Was war die größte absolute Änderung in der Matrix?
        max_diff = np.max(np.abs(current_matrix - prev_matrix))
        iteration += 1

    print(f"Konvergenz erreicht nach {iteration} Iterationen (Max Differenz: {max_diff:.2e})")

    ds_new = ds.copy(deep=True)

    # 2. Ersetze die Werte der Variable "network" durch unsere neue Matrix.
    # Wir übergeben die Dimensionen der alten Variable, damit xarray die Struktur korrekt zuordnet.
    ds_new["network"] = (ds["network"].dims, current_matrix)

    # 3. Automatischen Dateinamen generieren, falls kein Pfad angegeben wurde
    output_filepath = f"./pre-processed-data/water_BaU/scenario_ssp{scenario}_decade{year}_month{month:02d}_BaU.nc"

    # 4. Das modifizierte Dataset als neue NetCDF-Datei auf die Festplatte schreiben
    ds_new.to_netcdf(output_filepath)

    # 5. Dateien sauber schließen, um den Arbeitsspeicher freizugeben und File-Locks zu verhindern
    ds.close()
    ds_new.close()

    print(f"Datei erfolgreich erstellt und gespeichert unter: {output_filepath}")


def execute_deforestation():
    """Führt die Entwaldungsberechnungen für alle Szenarien, Jahre und Monate aus."""
    scenarios = [245, 370, 585]
    years_def = range(2030, 2051)
    years_raw = range(2030, 2100)
    months = range(1, 13)  # Alle Monate

    for scenario in scenarios:
        for year in years_def:
            for month in months:
                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (BaU)")
                deforestation_network_BaU_one_step(scenario, year, month)

                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (Gov)")
                deforestation_network_Gov_one_step(scenario, year, month)

        for year in years_raw:
            for month in months:
                print(f"Verarbeite Szenario {scenario}, Jahr {year}, Monat {month} (Sec)")
                deforestation_network_Sec(scenario, year, month)


if __name__ == "__main__":
    execute_deforestation()
