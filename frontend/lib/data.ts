import fs from "node:fs/promises";
import path from "node:path";

import { cache } from "react";
import { parse } from "csv-parse/sync";
import * as h3 from "h3-js";

type RawRow = Record<string, string>;

type LandmarkRow = {
  latitude: number;
  longitude: number;
  landmarkLabel: string;
  source: string;
};

type ZoneAccumulator = {
  displayZone: string;
  predictedCallsNext30d: number;
  codeViolationsCount: number;
  service311Count: number;
  poiCount: number;
  mostVisitedCount: number;
  mostVisitedTotalVisits: number;
  distanceToNearestStationSum: number;
  distanceToNearestStationCount: number;
  distanceToNearestPoiSum: number;
  distanceToNearestPoiCount: number;
  predictedCallsNext30dP10: number;
  predictedCallsNext30dP90: number;
  predictionUncertainty30d: number;
  interventionPriorityScore: number;
  demandTier: string;
};

export type ZoneBase = {
  h3Cell: string;
  centroidLatitude: number;
  centroidLongitude: number;
  predictedCallsNext30d: number;
  codeViolationsCount: number;
  service311Count: number;
  poiCount: number;
  mostVisitedCount: number;
  mostVisitedTotalVisits: number;
  distanceToNearestStationKm: number;
  distanceToNearestPoiKm: number;
  predictedCallsNext30dP10: number;
  predictedCallsNext30dP90: number;
  predictionUncertainty30d: number;
  interventionPriorityScore: number;
  demandTier: string;
  areaLabel: string;
};

export type TopPredictor = {
  feature: string;
  importance: number;
};

export type DatasetUsageRow = {
  dataset: string;
  status: "loaded" | "missing" | "read_error";
  spatialOrGeometry?: "yes" | "no";
};

export type BudgetSummary = {
  budgetUsd: number;
  actionsSelected: number;
  cellsTouched: number;
  spentUsd: number;
  budgetUtilizationPct: number;
  avgRoiCallsPreventedPer1k: number;
  preventedCalls30d: number;
  equitySpendSharePct: number;
};

export type PortfolioRow = {
  h3Cell: string;
  centroidLatitude: number;
  centroidLongitude: number;
  action: string;
  costUsd: number;
  reductionPct: number;
  baselineCalls30d: number;
  preventedCalls30d: number;
  roiCallsPreventedPer1k: number;
  codeViolationsCount: number;
  service311Count: number;
  distanceToNearestStationKm: number;
  predictionUncertainty30d: number;
  underservedFlag: boolean;
  priorityScore: number;
  budgetUsd: number;
  equitySpendSharePct: number;
  locationLabel: string;
};

export type AlertRow = {
  h3Cell: string;
  centroidLatitude: number;
  centroidLongitude: number;
  alertLevel: string;
  stressIndex: number;
  alertReason: string;
  predictedCallsNext30d: number;
  predictedCallsNext30dP90: number;
  codeViolationsCount: number;
  service311Count: number;
  predictionUncertainty30d: number;
  distanceToNearestStationKm: number;
  equityNeedScore?: number;
  locationLabel: string;
};

export type GovernanceSummary = {
  holdoutRmse?: number;
  cvMaeMean?: number;
  cvR2Mean?: number;
};

export type DashboardData = {
  dataDirName: string;
  generatedAtUtc: string;
  zoneBases: ZoneBase[];
  topPredictors: TopPredictor[];
  totalPredicted30d: number;
  totalUncertainty30d: number;
  uncertaintyPct: number;
  topDriver: string;
  targetSource: string;
  r2?: number;
  mae?: number;
  rmse?: number;
  moranI?: number;
  governance: GovernanceSummary;
  datasetUsage: DatasetUsageRow[];
  budgets: BudgetSummary[];
  portfolio: PortfolioRow[];
  alerts: AlertRow[];
  equityWatch: AlertRow[];
  pitchBrief: string;
};

const DATASET_USAGE_NAMES = [
  "911_calls",
  "code_violations",
  "paving_project",
  "fire_and_police_station",
  "city_limit",
  "received_311_service_requests",
  "point_of_interest",
  "most_visited_locations",
  "community_centers",
  "parks_and_trail",
  "education_facility",
  "pharmacy_locator"
];

function numberOr(value: unknown, fallback = 0): number {
  if (value === null || value === undefined) return fallback;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (text === "" || text === "nan" || text === "null" || text === "none") return fallback;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function numberOrNaN(value: unknown): number {
  if (value === null || value === undefined) return Number.NaN;
  if (typeof value === "string") {
    const text = value.trim().toLowerCase();
    if (text === "" || text === "nan" || text === "null" || text === "none") return Number.NaN;
  }
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function pickColumn(columns: string[], candidates: string[]): string | null {
  const lowerMap = new Map(columns.map((c) => [c.toLowerCase(), c]));
  for (const candidate of candidates) {
    const found = lowerMap.get(candidate.toLowerCase());
    if (found) return found;
  }
  return null;
}

function cleanKey(value: string): string {
  return value.trim().toUpperCase().replace(/\s+/g, " ");
}

function mergeLandmarkLabel(label: string, streetHint: string): string {
  if (!label) return streetHint;
  if (!streetHint) return label;

  const normLabel = cleanKey(label);
  const normStreet = cleanKey(streetHint);
  if (normLabel === normStreet || normLabel.includes(normStreet)) {
    return label;
  }
  return `${label} (${streetHint})`;
}

function extractLandmarks(rows: RawRow[], sourceTag: string): LandmarkRow[] {
  if (rows.length === 0) return [];

  const columns = Object.keys(rows[0] ?? {});
  const latCol = pickColumn(columns, ["latitude", "Latitude", "LATITUDE", "lat", "y"]);
  const lonCol = pickColumn(columns, ["longitude", "Longitude", "LONGITUDE", "lon", "x"]);
  if (!latCol || !lonCol) return [];

  const labelCol = pickColumn(columns, [
    "Place_Name",
    "place_name",
    "Location_Name",
    "location_name",
    "Facility_Name",
    "facility_name",
    "Name",
    "name",
    "Business_Name",
    "business_name",
    "FULLADDR",
    "fulladdr",
    "Address",
    "address",
    "Description",
    "description"
  ]);
  const streetCol = pickColumn(columns, [
    "FULLADDR",
    "fulladdr",
    "Address",
    "address",
    "Street",
    "street",
    "street_name",
    "Street_Name"
  ]);
  const typeCol = pickColumn(columns, ["Type", "type", "Category", "category", "Location_Category", "location_category"]);

  const out: LandmarkRow[] = [];
  for (const row of rows) {
    const latitude = numberOrNaN(row[latCol]);
    const longitude = numberOrNaN(row[lonCol]);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) continue;

    let label = labelCol ? String(row[labelCol] ?? "").trim() : "";
    const streetHint = streetCol ? String(row[streetCol] ?? "").trim() : "";
    const typeHint = typeCol ? String(row[typeCol] ?? "").trim() : "";

    if (typeHint && label) {
      label = `${typeHint} - ${label}`;
    } else if (typeHint && !label) {
      label = typeHint;
    }

    label = mergeLandmarkLabel(label, streetHint);
    if (!label || label.toLowerCase() === "nan") continue;

    out.push({
      latitude,
      longitude,
      landmarkLabel: label,
      source: sourceTag
    });
  }

  const seen = new Set<string>();
  const deduped: LandmarkRow[] = [];
  for (const row of out) {
    const key = `${row.latitude}|${row.longitude}|${row.landmarkLabel}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
  }
  return deduped;
}

async function pathExists(target: string): Promise<boolean> {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

async function resolveDataDir(): Promise<string> {
  const candidates = [
    path.resolve(process.cwd(), "data", "served"),
    path.resolve(process.cwd(), "..", "data", "served"),
    path.resolve(process.cwd(), "data", "processed"),
    path.resolve(process.cwd(), "..", "data", "processed")
  ];

  for (const candidate of candidates) {
    if (await pathExists(candidate)) {
      return candidate;
    }
  }

  return candidates[0];
}

async function readJson<T extends Record<string, unknown>>(dataDir: string, filename: string): Promise<T> {
  const filePath = path.join(dataDir, filename);
  if (!(await pathExists(filePath))) {
    return {} as T;
  }

  try {
    const content = await fs.readFile(filePath, "utf8");
    return JSON.parse(content) as T;
  } catch {
    return {} as T;
  }
}

async function readText(dataDir: string, filename: string): Promise<string> {
  const filePath = path.join(dataDir, filename);
  if (!(await pathExists(filePath))) {
    return "";
  }
  try {
    return await fs.readFile(filePath, "utf8");
  } catch {
    return "";
  }
}

async function readCsv(dataDir: string, filename: string, maxDataRows?: number): Promise<RawRow[]> {
  const filePath = path.join(dataDir, filename);
  if (!(await pathExists(filePath))) {
    return [];
  }

  try {
    const content = await fs.readFile(filePath, "utf8");
    const records = parse(content, {
      columns: true,
      skip_empty_lines: true,
      relax_column_count: true,
      trim: true,
      ...(typeof maxDataRows === "number" ? { to_line: maxDataRows + 1 } : {})
    }) as RawRow[];
    return records;
  } catch {
    return [];
  }
}

function displayParent(cell: string, levelsUp = 2): string {
  try {
    const resolution = h3.getResolution(cell);
    const targetResolution = Math.max(0, resolution - levelsUp);
    return h3.cellToParent(cell, targetResolution);
  } catch {
    return cell;
  }
}

function aggregateDisplayZones(predictions: RawRow[]): ZoneBase[] {
  const byZone = new Map<string, ZoneAccumulator>();

  for (const row of predictions) {
    const h3Cell = String(row.h3_cell ?? "").trim();
    if (!h3Cell) continue;

    const zone = displayParent(h3Cell, 2);
    const existing = byZone.get(zone) ?? {
      displayZone: zone,
      predictedCallsNext30d: 0,
      codeViolationsCount: 0,
      service311Count: 0,
      poiCount: 0,
      mostVisitedCount: 0,
      mostVisitedTotalVisits: 0,
      distanceToNearestStationSum: 0,
      distanceToNearestStationCount: 0,
      distanceToNearestPoiSum: 0,
      distanceToNearestPoiCount: 0,
      predictedCallsNext30dP10: 0,
      predictedCallsNext30dP90: 0,
      predictionUncertainty30d: 0,
      interventionPriorityScore: 0,
      demandTier: ""
    };

    existing.predictedCallsNext30d += numberOr(row.predicted_calls_next_30d);
    existing.codeViolationsCount += numberOr(row.code_violations_count);
    existing.service311Count += numberOr(row.service_311_count);
    existing.poiCount += numberOr(row.poi_count);
    existing.mostVisitedCount += numberOr(row.most_visited_count);
    existing.mostVisitedTotalVisits += numberOr(row.most_visited_total_visits);
    existing.predictedCallsNext30dP10 += numberOr(row.predicted_calls_next_30d_p10);
    existing.predictedCallsNext30dP90 += numberOr(row.predicted_calls_next_30d_p90);
    existing.predictionUncertainty30d += numberOr(row.prediction_uncertainty_30d);
    existing.interventionPriorityScore += numberOr(row.intervention_priority_score);

    const stationDist = numberOrNaN(row.distance_to_nearest_station_km);
    if (Number.isFinite(stationDist)) {
      existing.distanceToNearestStationSum += stationDist;
      existing.distanceToNearestStationCount += 1;
    }

    const poiDist = numberOrNaN(row.distance_to_nearest_poi_km);
    if (Number.isFinite(poiDist)) {
      existing.distanceToNearestPoiSum += poiDist;
      existing.distanceToNearestPoiCount += 1;
    }

    if (!existing.demandTier) {
      existing.demandTier = String(row.demand_tier ?? "");
    }

    byZone.set(zone, existing);
  }

  const zones: ZoneBase[] = [];
  for (const [, value] of byZone) {
    let centroidLatitude = 0;
    let centroidLongitude = 0;
    try {
      const [lat, lon] = h3.cellToLatLng(value.displayZone);
      centroidLatitude = lat;
      centroidLongitude = lon;
    } catch {
      centroidLatitude = 0;
      centroidLongitude = 0;
    }

    zones.push({
      h3Cell: value.displayZone,
      centroidLatitude,
      centroidLongitude,
      predictedCallsNext30d: value.predictedCallsNext30d,
      codeViolationsCount: value.codeViolationsCount,
      service311Count: value.service311Count,
      poiCount: value.poiCount,
      mostVisitedCount: value.mostVisitedCount,
      mostVisitedTotalVisits: value.mostVisitedTotalVisits,
      distanceToNearestStationKm:
        value.distanceToNearestStationCount > 0
          ? value.distanceToNearestStationSum / value.distanceToNearestStationCount
          : Number.NaN,
      distanceToNearestPoiKm:
        value.distanceToNearestPoiCount > 0 ? value.distanceToNearestPoiSum / value.distanceToNearestPoiCount : Number.NaN,
      predictedCallsNext30dP10: value.predictedCallsNext30dP10,
      predictedCallsNext30dP90: value.predictedCallsNext30dP90,
      predictionUncertainty30d: value.predictionUncertainty30d,
      interventionPriorityScore: value.interventionPriorityScore,
      demandTier: value.demandTier,
      areaLabel: ""
    });
  }

  return zones;
}

function nearestLandmarkLabel(lat: number, lon: number, lookup: LandmarkRow[]): string {
  if (lookup.length === 0) {
    return `Near (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
  }

  let nearest = lookup[0];
  let bestDistance = Number.POSITIVE_INFINITY;

  for (const lm of lookup) {
    const d2 = (lm.latitude - lat) ** 2 + (lm.longitude - lon) ** 2;
    if (d2 < bestDistance) {
      bestDistance = d2;
      nearest = lm;
    }
  }

  const name = String(nearest.landmarkLabel ?? "").trim();
  if (!name) {
    return `Near (${lat.toFixed(4)}, ${lon.toFixed(4)})`;
  }
  return `Near ${name.slice(0, 56)}`;
}

function friendlyLabels(
  points: Array<{ centroidLatitude: number; centroidLongitude: number }>,
  lookup: LandmarkRow[],
  ensureUnique = true
): string[] {
  const labels: string[] = [];
  const used = new Map<string, number>();

  for (const point of points) {
    const base = nearestLandmarkLabel(point.centroidLatitude, point.centroidLongitude, lookup);
    if (!ensureUnique) {
      labels.push(base);
      continue;
    }
    const count = (used.get(base) ?? 0) + 1;
    used.set(base, count);
    labels.push(count > 1 ? `${base} (${count})` : base);
  }

  return labels;
}

function parseBoolean(value: unknown): boolean {
  const normalized = String(value ?? "").trim().toLowerCase();
  return normalized === "true" || normalized === "1" || normalized === "yes";
}

async function loadLandmarkLookup(dataDir: string): Promise<LandmarkRow[]> {
  const files: Array<[string, string]> = [
    ["most_visited_locations.csv", "visited"],
    ["point_of_interest.csv", "poi"],
    ["community_centers.csv", "community"],
    ["parks_and_trail.csv", "parks"],
    ["education_facility.csv", "education"],
    ["pharmacy_locator.csv", "pharmacy"],
    ["business_license.csv", "business"],
    ["received_311_service_requests.csv", "service"]
  ];

  const parts: LandmarkRow[] = [];
  for (const [filename, tag] of files) {
    const rows = await readCsv(dataDir, filename);
    if (rows.length === 0) continue;
    parts.push(...extractLandmarks(rows, tag));
  }

  const seen = new Set<string>();
  const deduped: LandmarkRow[] = [];
  for (const row of parts) {
    const key = `${row.latitude}|${row.longitude}|${row.landmarkLabel}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(row);
  }
  return deduped;
}

async function datasetUsageSummary(dataDir: string): Promise<DatasetUsageRow[]> {
  const rows: DatasetUsageRow[] = [];
  for (const name of DATASET_USAGE_NAMES) {
    const filePath = path.join(dataDir, `${name}.csv`);
    if (!(await pathExists(filePath))) {
      rows.push({ dataset: name, status: "missing" });
      continue;
    }

    const sample = await readCsv(dataDir, `${name}.csv`, 1);
    if (sample.length === 0) {
      rows.push({ dataset: name, status: "read_error" });
      continue;
    }

    const columns = Object.keys(sample[0] ?? {});
    const hasSpatial = columns.some((col) => {
      const c = col.toLowerCase();
      return c === "latitude" || c === "longitude" || c === "geometry_json";
    });

    rows.push({
      dataset: name,
      status: "loaded",
      spatialOrGeometry: hasSpatial ? "yes" : "no"
    });
  }
  return rows;
}

function toBudgetSummary(row: Record<string, unknown>): BudgetSummary {
  return {
    budgetUsd: numberOr(row.budget_usd),
    actionsSelected: numberOr(row.actions_selected),
    cellsTouched: numberOr(row.cells_touched),
    spentUsd: numberOr(row.spent_usd),
    budgetUtilizationPct: numberOr(row.budget_utilization_pct),
    avgRoiCallsPreventedPer1k: numberOr(row.avg_roi_calls_prevented_per_1k),
    preventedCalls30d: numberOr(row.prevented_calls_30d),
    equitySpendSharePct: numberOr(row.equity_spend_share_pct)
  };
}

export const getDashboardData = cache(async (): Promise<DashboardData> => {
  const dataDir = await resolveDataDir();

  const [
    predictions,
    topPredictorsRows,
    metadata,
    metrics,
    moran,
    governance,
    opsSummary,
    portfolioRows,
    alertsRows,
    equityRows,
    pitchBrief,
    landmarkLookup,
    datasetUsage
  ] = await Promise.all([
    readCsv(dataDir, "predictions.csv"),
    readCsv(dataDir, "top_predictors.csv"),
    readJson<Record<string, unknown>>(dataDir, "metadata.json"),
    readJson<Record<string, unknown>>(dataDir, "model_metrics.json"),
    readJson<Record<string, unknown>>(dataDir, "bivariate_moran_summary.json"),
    readJson<Record<string, unknown>>(dataDir, "model_governance.json"),
    readJson<Record<string, unknown>>(dataDir, "operations_summary.json"),
    readCsv(dataDir, "intervention_portfolio.csv"),
    readCsv(dataDir, "early_warning_alerts.csv"),
    readCsv(dataDir, "equity_watchlist.csv"),
    readText(dataDir, "pitch_summary.md"),
    loadLandmarkLookup(dataDir),
    datasetUsageSummary(dataDir)
  ]);

  const zones = aggregateDisplayZones(predictions);
  const zoneLabels = friendlyLabels(
    zones.map((zone) => ({
      centroidLatitude: zone.centroidLatitude,
      centroidLongitude: zone.centroidLongitude
    })),
    landmarkLookup,
    true
  );

  const zoneBases: ZoneBase[] = zones.map((zone, idx) => ({
    ...zone,
    areaLabel: zoneLabels[idx] ?? `Near (${zone.centroidLatitude.toFixed(4)}, ${zone.centroidLongitude.toFixed(4)})`
  }));

  const totalPredicted30d = zoneBases.reduce((sum, row) => sum + row.predictedCallsNext30d, 0);
  const totalUncertainty30d = zoneBases.reduce((sum, row) => sum + row.predictionUncertainty30d, 0);
  const uncertaintyPct = (totalUncertainty30d / Math.max(totalPredicted30d, 1e-9)) * 100;

  const topPredictors: TopPredictor[] = topPredictorsRows.map((row) => ({
    feature: String(row.feature ?? ""),
    importance: numberOr(row.importance)
  }));

  const portfolio: PortfolioRow[] = portfolioRows.map((row) => {
    const centroidLatitude = numberOr(row.centroid_latitude);
    const centroidLongitude = numberOr(row.centroid_longitude);
    return {
      h3Cell: String(row.h3_cell ?? ""),
      centroidLatitude,
      centroidLongitude,
      action: String(row.action ?? "Action"),
      costUsd: numberOr(row.cost_usd),
      reductionPct: numberOr(row.reduction_pct),
      baselineCalls30d: numberOr(row.baseline_calls_30d),
      preventedCalls30d: numberOr(row.prevented_calls_30d),
      roiCallsPreventedPer1k: numberOr(row.roi_calls_prevented_per_1k),
      codeViolationsCount: numberOr(row.code_violations_count),
      service311Count: numberOr(row.service_311_count),
      distanceToNearestStationKm: numberOr(row.distance_to_nearest_station_km, Number.NaN),
      predictionUncertainty30d: numberOr(row.prediction_uncertainty_30d),
      underservedFlag: parseBoolean(row.underserved_flag),
      priorityScore: numberOr(row.priority_score),
      budgetUsd: numberOr(row.budget_usd),
      equitySpendSharePct: numberOr(row.equity_spend_share_pct),
      locationLabel: nearestLandmarkLabel(centroidLatitude, centroidLongitude, landmarkLookup)
    };
  });

  const toAlertRow = (row: RawRow): AlertRow => {
    const centroidLatitude = numberOr(row.centroid_latitude);
    const centroidLongitude = numberOr(row.centroid_longitude);
    return {
      h3Cell: String(row.h3_cell ?? ""),
      centroidLatitude,
      centroidLongitude,
      alertLevel: String(row.alert_level ?? "watch"),
      stressIndex: numberOr(row.stress_index),
      alertReason: String(row.alert_reason ?? "stacked risk factors"),
      predictedCallsNext30d: numberOr(row.predicted_calls_next_30d),
      predictedCallsNext30dP90: numberOr(row.predicted_calls_next_30d_p90),
      codeViolationsCount: numberOr(row.code_violations_count),
      service311Count: numberOr(row.service_311_count),
      predictionUncertainty30d: numberOr(row.prediction_uncertainty_30d),
      distanceToNearestStationKm: numberOr(row.distance_to_nearest_station_km, Number.NaN),
      equityNeedScore: row.equity_need_score === undefined ? undefined : numberOr(row.equity_need_score),
      locationLabel: nearestLandmarkLabel(centroidLatitude, centroidLongitude, landmarkLookup)
    };
  };

  const alerts = alertsRows.map(toAlertRow).sort((a, b) => b.stressIndex - a.stressIndex);
  const equityWatch = equityRows.map(toAlertRow).sort((a, b) => b.stressIndex - a.stressIndex);

  const summaryRows = Array.isArray(opsSummary.portfolio_summary)
    ? (opsSummary.portfolio_summary as Array<Record<string, unknown>>)
    : [];
  const budgets = summaryRows.map(toBudgetSummary).sort((a, b) => a.budgetUsd - b.budgetUsd);

  const holdout = (governance.evaluation as Record<string, unknown> | undefined)?.holdout as
    | Record<string, unknown>
    | undefined;
  const cv = (governance.evaluation as Record<string, unknown> | undefined)?.cross_validation as
    | Record<string, unknown>
    | undefined;
  const cvEnsemble = cv?.ensemble as Record<string, unknown> | undefined;

  return {
    dataDirName: path.basename(dataDir),
    generatedAtUtc: String(metadata.generated_at_utc ?? ""),
    zoneBases,
    topPredictors,
    totalPredicted30d,
    totalUncertainty30d,
    uncertaintyPct,
    topDriver: String(topPredictors[0]?.feature ?? "n/a"),
    targetSource: String(metrics.target_source ?? "unknown"),
    r2: metrics.r2 === undefined ? undefined : numberOr(metrics.r2),
    mae: metrics.mae === undefined ? undefined : numberOr(metrics.mae),
    rmse: metrics.rmse === undefined ? undefined : numberOr(metrics.rmse),
    moranI: moran.bivariate_moran_i === undefined ? undefined : numberOr(moran.bivariate_moran_i),
    governance: {
      holdoutRmse: holdout?.rmse === undefined ? undefined : numberOr(holdout.rmse),
      cvMaeMean: cvEnsemble?.mae_mean === undefined ? undefined : numberOr(cvEnsemble.mae_mean),
      cvR2Mean: cvEnsemble?.r2_mean === undefined ? undefined : numberOr(cvEnsemble.r2_mean)
    },
    datasetUsage,
    budgets,
    portfolio,
    alerts,
    equityWatch,
    pitchBrief
  };
});
