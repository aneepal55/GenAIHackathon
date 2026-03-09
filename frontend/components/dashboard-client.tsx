"use client";

import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Bar, BarChart, CartesianGrid, LabelList, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

import type { AlertRow, DashboardData, PortfolioRow, ZoneBase } from "../lib/data";
import type { MapZone } from "./risk-map";

type Props = {
  data: DashboardData;
};

type ZoneWithRisk = ZoneBase & {
  riskToday: number;
  color: [number, number, number, number];
  elevation: number;
  chanceToday: string;
  forecastTodayCount: string;
  forecastTodayText: string;
  hiddenRiskFlag: string;
  simulated30d: number;
  simulatedToday: number;
  preventable30d: number;
};

type TabKey = "overview" | "innovation" | "budget" | "operations" | "alerts" | "pitch";

type TableRow = Record<string, string>;

type TabItem = {
  key: TabKey;
  label: string;
  hint: string;
};

type PitchSection = {
  title: string;
  bullets: string[];
  paragraphs: string[];
};

type PitchParsed = {
  title: string;
  lead: string[];
  sections: PitchSection[];
};

const NAV_TABS: TabItem[] = [
  { key: "overview", label: "Overview", hint: "Map + Snapshot" },
  { key: "innovation", label: "Innovation", hint: "Scenario Lab" },
  { key: "budget", label: "Budget", hint: "Optimizer" },
  { key: "operations", label: "Operations", hint: "Action Views" },
  { key: "alerts", label: "Alerts", hint: "Warning + Equity" },
  { key: "pitch", label: "Pitch", hint: "Narrative" }
];

const RiskMap = dynamic(() => import("./risk-map"), { ssr: false });

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor(q * (sorted.length - 1))));
  return sorted[idx] ?? 0;
}

function chanceLabel(value: number, lowCut: number, highCut: number): string {
  if (value >= highCut) return "High chance today";
  if (value >= lowCut) return "Moderate chance today";
  return "Low chance today";
}

function formatRequestCount(value: number): string {
  return value < 0.5 ? "<1" : String(Math.round(value));
}

function requestsPhrase(value: number): string {
  if (value < 0.5) return "Likely quiet today (0 to 1 request)";
  if (value < 1.5) return "Likely around 1 request today";
  if (value < 3.5) return "Likely around 2 to 3 requests today";
  return `Likely around ${Math.round(value)} requests today`;
}

function formatKm(value: number, allowZero = true): string {
  if (!Number.isFinite(value)) return "Not available";
  if (!allowZero && value <= 0) return "Not available";
  return value.toFixed(2);
}

function colorForRisk(values: number[]): Array<[number, number, number, number]> {
  if (values.length === 0) return [];

  const sortedIndexes = values.map((_, idx) => idx).sort((a, b) => values[a] - values[b]);
  const pct = new Array<number>(values.length).fill(0);
  const n = values.length;

  sortedIndexes.forEach((idx, rank) => {
    pct[idx] = (rank + 1) / n;
  });

  return pct.map((normalized) => {
    const red = Math.round(25 + normalized * 230);
    const green = Math.round(90 + (1 - normalized) * 90);
    const blue = Math.round(255 - normalized * 240);
    return [red, green, blue, 210];
  });
}

function riskByDay(row: ZoneBase, dayIndex: number): number {
  const phase = (2 * Math.PI * dayIndex) / 30.0;
  const lonPhase = (row.centroidLongitude * Math.PI) / 180.0;
  const latPhase = (row.centroidLatitude * Math.PI) / 180.0;
  const wave = 0.75 + (dayIndex / 30.0) * 0.5 + 0.45 * Math.sin(phase + lonPhase + 0.4 * latPhase);
  const dailyBase = row.predictedCallsNext30d / 30.0;
  return Math.max(0, dailyBase * wave);
}

function interventionMultiplier(codeCleanupPct: number, rapidResponsePct: number, hotspotPatrolPct: number): number {
  let remaining = 1.0;
  remaining *= 1.0 - (codeCleanupPct / 100.0) * 0.35;
  remaining *= 1.0 - (rapidResponsePct / 100.0) * 0.25;
  remaining *= 1.0 - (hotspotPatrolPct / 100.0) * 0.2;
  return Math.max(0.55, remaining);
}

function dailyProfile(predicted30d: number, lat: number, lon: number, multiplier = 1.0): number[] {
  const out: number[] = [];
  for (let dayIndex = 0; dayIndex <= 30; dayIndex += 1) {
    const phase = (2 * Math.PI * dayIndex) / 30.0;
    const lonPhase = (lon * Math.PI) / 180.0;
    const latPhase = (lat * Math.PI) / 180.0;
    const wave = 0.75 + (dayIndex / 30.0) * 0.5 + 0.45 * Math.sin(phase + lonPhase + 0.4 * latPhase);
    const daily = Math.max(0, (predicted30d / 30.0) * wave * multiplier);
    out.push(daily);
  }
  return out;
}

function hiddenRiskFlag(row: ZoneWithRisk, codeQ90: number, reqQ50: number): string {
  if (row.codeViolationsCount >= codeQ90 && row.service311Count <= reqQ50 && row.chanceToday !== "High chance today") {
    return "Potential blind spot";
  }
  return "None";
}

function whyFlagged(row: ZoneWithRisk): string {
  const reasons: string[] = [];
  if (row.codeViolationsCount >= 20) reasons.push("many open code complaints");
  if (row.service311Count >= 500) reasons.push("high past 311 service demand");
  if (row.mostVisitedTotalVisits >= 500) reasons.push("high visitor activity nearby");
  if (row.distanceToNearestStationKm >= 3.0) reasons.push("farther from response station");
  if (reasons.length === 0) reasons.push("combined demand indicators are elevated");
  return reasons.join(", ");
}

function roiText(row: ZoneWithRisk): string {
  const violations = row.codeViolationsCount;
  const service311 = row.service311Count;
  const poiCount = row.poiCount;
  const predicted30 = row.predictedCallsNext30d;
  const stationDistance = row.distanceToNearestStationKm;
  const poiDistance = row.distanceToNearestPoiKm;

  const intensity = Math.log1p(Math.max(0, violations)) * 1.6 + Math.log1p(Math.max(0, service311)) * 0.8;
  const poiPressure = Math.min(2.0, Math.log1p(Math.max(0, poiCount)) * 0.5);
  const potentialReductionPct = Math.min(24.0, 2.5 + intensity + poiPressure);
  const expectedCallsReduced = predicted30 * (potentialReductionPct / 100.0);
  const focusCount = Math.max(3, Math.round(Math.min(80, Math.sqrt(Math.max(0, violations)) * 4)));
  const stationText = !Number.isFinite(stationDistance) || stationDistance <= 0 ? "not available" : `${stationDistance.toFixed(2)} km`;

  return (
    `If the city resolves about ${focusCount} high-priority nuisance/code issues in this area, service demand may drop by about ` +
    `${potentialReductionPct.toFixed(1)}% (~${expectedCallsReduced.toFixed(1)} fewer requests this month). ` +
    `Nearest station distance: ${stationText}. Nearest POI distance: ${poiDistance.toFixed(2)} km.`
  );
}

function actionPlanText(
  selectedArea: string,
  selectedRow: ZoneWithRisk,
  chanceToday: string,
  baseline30d: number,
  adjusted30d: number
): string {
  const reduction = Math.max(0, baseline30d - adjusted30d);
  return [
    `# Action Plan: ${selectedArea}`,
    "",
    `- Current risk signal: **${chanceToday}**`,
    `- Baseline expected requests (30d): **${baseline30d.toFixed(0)}**`,
    `- Simulated expected requests after interventions (30d): **${adjusted30d.toFixed(0)}**`,
    `- Simulated reduction: **${reduction.toFixed(0)} requests**`,
    "",
    "## Why this area",
    `- Open neighborhood code complaints: ${selectedRow.codeViolationsCount.toFixed(0)}`,
    `- Past city service requests (311): ${selectedRow.service311Count.toFixed(0)}`,
    `- Places/POIs in this area: ${selectedRow.poiCount.toFixed(0)}`,
    "",
    "## Recommended 2-week playbook",
    "1. Run focused code-enforcement sweep on top nuisance blocks.",
    "2. Dispatch proactive cleanup for illegal dumping/trash hotspots.",
    "3. Add temporary hotspot patrol windows during peak demand hours.",
    "4. Re-evaluate this zone in 7 days and adjust resources.",
    ""
  ].join("\n");
}

function ensureUniqueLabels(values: string[]): string[] {
  const used = new Map<string, number>();
  return values.map((value) => {
    const count = (used.get(value) ?? 0) + 1;
    used.set(value, count);
    return count > 1 ? `${value} (${count})` : value;
  });
}

function downloadText(filename: string, text: string, mime = "text/plain") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function compact(value: number): string {
  if (!Number.isFinite(value)) return "0";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

function valueText(value: number | undefined, digits = 2): string {
  if (value === undefined || !Number.isFinite(value)) return "n/a";
  return value.toFixed(digits);
}

function stripMarkdown(text: string): string {
  return text.replace(/\*\*/g, "").replace(/`/g, "").trim();
}

function parsePitchBrief(brief: string): PitchParsed {
  const lines = brief.split(/\r?\n/).map((line) => line.trim());
  let title = "Pitch Brief";
  const lead: string[] = [];
  const sections: PitchSection[] = [];
  let current: PitchSection | null = null;

  for (const raw of lines) {
    if (!raw) continue;

    if (raw.startsWith("# ")) {
      title = stripMarkdown(raw.slice(2));
      continue;
    }

    if (raw.startsWith("## ")) {
      if (current) sections.push(current);
      current = { title: stripMarkdown(raw.slice(3)), bullets: [], paragraphs: [] };
      continue;
    }

    const isBullet = raw.startsWith("- ");
    const cleaned = stripMarkdown(isBullet ? raw.slice(2) : raw);
    if (!cleaned) continue;

    if (current) {
      if (isBullet) current.bullets.push(cleaned);
      else current.paragraphs.push(cleaned);
    } else {
      lead.push(cleaned);
    }
  }

  if (current) sections.push(current);
  return { title, lead, sections };
}

function formatShortDate(isoDate: string): string {
  const parsed = new Date(isoDate);
  if (Number.isNaN(parsed.getTime())) return isoDate;
  return parsed.toLocaleDateString("en-US", { month: "short", day: "numeric" });
}

function formatRequestsWithContext(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return "0 expected requests";
  if (value < 0.5) return "<1 expected request";
  if (value < 1.5) return "About 1 expected request";
  return `About ${Math.round(value)} expected requests`;
}

function DataTable({ columns, rows }: { columns: string[]; rows: TableRow[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={`${idx}-${row[columns[0]] ?? "row"}`}>
              {columns.map((column) => (
                <td key={`${idx}-${column}`}>{row[column] ?? ""}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function alertLevelCounts(alerts: AlertRow[]) {
  const counts = new Map<string, number>();
  for (const row of alerts) {
    counts.set(row.alertLevel, (counts.get(row.alertLevel) ?? 0) + 1);
  }
  return Array.from(counts.entries()).map(([level, count]) => ({ level, count }));
}

function actionMixRows(rows: PortfolioRow[]) {
  const grouped = new Map<string, number>();
  for (const row of rows) {
    grouped.set(row.action, (grouped.get(row.action) ?? 0) + row.preventedCalls30d);
  }
  return Array.from(grouped.entries())
    .map(([action, prevented]) => ({ action, prevented }))
    .sort((a, b) => b.prevented - a.prevented);
}

export default function DashboardClient({ data }: Props) {
  const [activeTab, setActiveTab] = useState<TabKey>("overview");
  const [showQuickGuide, setShowQuickGuide] = useState(true);
  const [timelineDay, setTimelineDay] = useState(0);
  const [selectedH3Cell, setSelectedH3Cell] = useState("");
  const [codeCleanupPct, setCodeCleanupPct] = useState(35);
  const [rapidResponsePct, setRapidResponsePct] = useState(30);
  const [hotspotPatrolPct, setHotspotPatrolPct] = useState(25);
  const [eventSurgePct, setEventSurgePct] = useState(10);
  const [budgetChoice, setBudgetChoice] = useState<number | null>(null);
  const [pitchVisible, setPitchVisible] = useState(false);

  const today = useMemo(() => {
    const d = new Date();
    d.setHours(0, 0, 0, 0);
    return d;
  }, []);

  const selectedDate = useMemo(() => {
    const d = new Date(today);
    d.setDate(today.getDate() + timelineDay);
    return d;
  }, [today, timelineDay]);

  const selectedDateIso = selectedDate.toISOString().slice(0, 10);
  const generatedDateOnly = useMemo(() => {
    if (!data.generatedAtUtc) return "";
    const parsed = new Date(data.generatedAtUtc);
    if (!Number.isNaN(parsed.getTime())) {
      return parsed.toISOString().slice(0, 10);
    }
    return String(data.generatedAtUtc).slice(0, 10);
  }, [data.generatedAtUtc]);

  const interventionMult = useMemo(
    () => interventionMultiplier(codeCleanupPct, rapidResponsePct, hotspotPatrolPct),
    [codeCleanupPct, rapidResponsePct, hotspotPatrolPct]
  );
  const eventMult = useMemo(() => 1.0 + (eventSurgePct / 100.0) * 0.45, [eventSurgePct]);
  const scenarioMult = interventionMult * eventMult;

  const baseRisk = useMemo(() => {
    const rows = data.zoneBases.map((row) => ({
      ...row,
      riskToday: riskByDay(row, timelineDay)
    }));

    const riskValues = rows.map((row) => row.riskToday);
    const lowCut = quantile(riskValues, 0.75);
    const highCut = quantile(riskValues, 0.95);
    const colors = colorForRisk(riskValues);
    const codeQ90 = quantile(rows.map((row) => row.codeViolationsCount), 0.9);
    const reqQ50 = quantile(rows.map((row) => row.service311Count), 0.5);

    const withFlags: ZoneWithRisk[] = rows.map((row, idx) => {
      const chanceToday = chanceLabel(row.riskToday, lowCut, highCut);
      const derived: ZoneWithRisk = {
        ...row,
        color: colors[idx] ?? [25, 180, 255, 210],
        elevation: Math.max(0, row.riskToday) * 350 + 10,
        chanceToday,
        forecastTodayCount: formatRequestCount(row.riskToday),
        forecastTodayText: requestsPhrase(row.riskToday),
        hiddenRiskFlag: "None",
        simulated30d: 0,
        simulatedToday: 0,
        preventable30d: 0
      };
      return derived;
    });

    const withHiddenRisk = withFlags.map((row) => ({
      ...row,
      hiddenRiskFlag: hiddenRiskFlag(row, codeQ90, reqQ50)
    }));

    return {
      rows: withHiddenRisk,
      lowCut,
      highCut
    };
  }, [data.zoneBases, timelineDay]);

  const zones = useMemo<ZoneWithRisk[]>(() => {
    return baseRisk.rows.map((row) => {
      const simulated30d = row.predictedCallsNext30d * scenarioMult;
      return {
        ...row,
        simulated30d,
        simulatedToday: row.riskToday * scenarioMult,
        preventable30d: Math.max(0, row.predictedCallsNext30d - simulated30d)
      };
    });
  }, [baseRisk.rows, scenarioMult]);

  const totalPredicted30d = data.totalPredicted30d;
  const forecastDayTotal = useMemo(() => zones.reduce((sum, row) => sum + row.riskToday, 0), [zones]);
  const highRiskCount = useMemo(() => {
    const totalRisk = zones.reduce((sum, row) => sum + row.riskToday, 0);
    if (totalRisk === 0) return 0;
    const threshold = quantile(
      zones.map((row) => row.riskToday),
      0.9
    );
    return zones.filter((row) => row.riskToday >= threshold).length;
  }, [zones]);

  const topCells = useMemo(() => {
    const topN = Math.min(20, zones.length);
    return [...zones]
      .sort((a, b) => b.riskToday - a.riskToday)
      .slice(0, topN)
      .map((row, idx) => ({
        ...row,
        areaLabelDisplay: `${row.areaLabel} (Top ${idx + 1})`
      }));
  }, [zones]);

  useEffect(() => {
    if (topCells.length === 0) return;
    if (!selectedH3Cell || !topCells.some((row) => row.h3Cell === selectedH3Cell)) {
      setSelectedH3Cell(topCells[0].h3Cell);
    }
  }, [selectedH3Cell, topCells]);

  const selectedRow = useMemo(() => {
    if (topCells.length === 0) return zones[0];
    return topCells.find((row) => row.h3Cell === selectedH3Cell) ?? topCells[0];
  }, [selectedH3Cell, topCells, zones]);

  const budgetOptions = useMemo(() => {
    const unique = new Set<number>();
    for (const row of data.portfolio) {
      if (Number.isFinite(row.budgetUsd)) unique.add(row.budgetUsd);
    }
    return Array.from(unique).sort((a, b) => a - b);
  }, [data.portfolio]);

  const defaultBudget = useMemo(() => {
    if (budgetOptions.length === 0) return 0;
    return budgetOptions[Math.min(budgetOptions.length - 1, 1)] ?? budgetOptions[0] ?? 0;
  }, [budgetOptions]);

  useEffect(() => {
    if (budgetOptions.length === 0) return;
    if (budgetChoice === null || !budgetOptions.includes(budgetChoice)) {
      setBudgetChoice(defaultBudget);
    }
  }, [budgetChoice, budgetOptions, defaultBudget]);

  const effectiveBudget = budgetChoice ?? defaultBudget;

  const portfolioChoice = useMemo(() => {
    if (data.portfolio.length === 0) return [];
    return data.portfolio
      .filter((row) => row.budgetUsd === effectiveBudget)
      .sort((a, b) => b.priorityScore - a.priorityScore)
      .slice(0, 20);
  }, [data.portfolio, effectiveBudget]);

  const budgetInfo = useMemo(() => data.budgets.find((row) => row.budgetUsd === effectiveBudget), [data.budgets, effectiveBudget]);

  const selectedBaseline30d = selectedRow?.predictedCallsNext30d ?? 0;
  const selectedAdjusted30d = selectedBaseline30d * scenarioMult;
  const selectedChance = selectedRow
    ? chanceLabel(
        selectedRow.riskToday * scenarioMult,
        baseRisk.lowCut * Math.max(scenarioMult, 1e-9),
        baseRisk.highCut * Math.max(scenarioMult, 1e-9)
      )
    : "Low chance today";

  const baselineCurve = selectedRow
    ? dailyProfile(selectedBaseline30d, selectedRow.centroidLatitude, selectedRow.centroidLongitude, 1)
    : [];
  const scenarioCurve = selectedRow
    ? dailyProfile(selectedBaseline30d, selectedRow.centroidLatitude, selectedRow.centroidLongitude, scenarioMult)
    : [];

  const trendSeries = baselineCurve.map((baseline, idx) => {
    const d = new Date(selectedDate);
    d.setDate(selectedDate.getDate() + idx);
    const iso = d.toISOString().slice(0, 10);
    return {
      dayIndex: idx,
      date: iso,
      dateLabel: formatShortDate(iso),
      baseline,
      scenario: scenarioCurve[idx] ?? 0
    };
  });

  const mapZones: MapZone[] = zones.map((row) => ({
    h3Cell: row.h3Cell,
    areaLabel: row.areaLabel,
    centroidLatitude: row.centroidLatitude,
    centroidLongitude: row.centroidLongitude,
    predictedCallsNext30d: row.predictedCallsNext30d,
    predictedCallsNext30dP10: row.predictedCallsNext30dP10,
    predictedCallsNext30dP90: row.predictedCallsNext30dP90,
    predictionUncertainty30d: row.predictionUncertainty30d,
    codeViolationsCount: row.codeViolationsCount,
    service311Count: row.service311Count,
    riskToday: row.riskToday,
    forecastTodayCount: row.forecastTodayCount,
    forecastTodayText: row.forecastTodayText,
    chanceToday: row.chanceToday,
    distanceToNearestStationKm: row.distanceToNearestStationKm,
    distanceToNearestPoiKm: row.distanceToNearestPoiKm,
    color: row.color
  }));

  const topAreas = useMemo(() => {
    const topRaw = [...zones].sort((a, b) => b.riskToday - a.riskToday).slice(0, 10);
    return topRaw.map((row, idx) => {
      let action = "Monitor";
      if (idx < 3) action = "Act now";
      else if (idx < 7) action = "Plan this week";

      return {
        area: row.areaLabel,
        recommendedAction: action,
        chance: row.chanceToday,
        expected30d: formatRequestCount(row.predictedCallsNext30d),
        codeComplaints: row.codeViolationsCount.toFixed(0),
        service311: row.service311Count.toFixed(0),
        blindSpot: row.hiddenRiskFlag,
        why: whyFlagged(row)
      };
    });
  }, [zones]);

  const leaderboard = useMemo(() => {
    const rows = [...zones].sort((a, b) => b.preventable30d - a.preventable30d).slice(0, 10);
    const maxPreventable = Math.max(1, ...rows.map((row) => row.preventable30d));
    return rows.map((row) => ({
      location: row.areaLabel,
      baseline30d: row.predictedCallsNext30d.toFixed(0),
      scenario30d: row.simulated30d.toFixed(0),
      preventable30d: row.preventable30d.toFixed(0),
      blindSpot: row.hiddenRiskFlag,
      barPct: (row.preventable30d / maxPreventable) * 100
    }));
  }, [zones]);

  const reviewQueue = useMemo(() => {
    return [...zones]
      .sort((a, b) => b.predictionUncertainty30d - a.predictionUncertainty30d)
      .slice(0, 8)
      .map((row) => ({
        location: row.areaLabel,
        spread30d: row.predictionUncertainty30d.toFixed(0),
        lower30d: row.predictedCallsNext30dP10.toFixed(0),
        upper30d: row.predictedCallsNext30dP90.toFixed(0),
        codeComplaints: row.codeViolationsCount.toFixed(0),
        service311: row.service311Count.toFixed(0),
        action: "Send field verification + refresh data"
      }));
  }, [zones]);

  const alerts = useMemo(() => {
    const src = data.alerts.slice(0, 12);
    const labels = ensureUniqueLabels(src.map((row) => row.locationLabel));
    return src.map((row, idx) => ({
      location: labels[idx] ?? row.locationLabel,
      level: row.alertLevel,
      reason: row.alertReason,
      expected30d: row.predictedCallsNext30d.toFixed(1),
      highEnd30d: row.predictedCallsNext30dP90.toFixed(1),
      stress: row.stressIndex.toFixed(1)
    }));
  }, [data.alerts]);

  const equity = useMemo(() => {
    const src = data.equityWatch.slice(0, 10);
    const labels = ensureUniqueLabels(src.map((row) => row.locationLabel));
    return src.map((row, idx) => ({
      location: labels[idx] ?? row.locationLabel,
      risk: row.alertLevel,
      stationKm: Number.isFinite(row.distanceToNearestStationKm) ? row.distanceToNearestStationKm.toFixed(2) : "Not available",
      stress: row.stressIndex.toFixed(2),
      concern: row.alertReason
    }));
  }, [data.equityWatch]);
  const hasEquityStationDistance = useMemo(
    () => data.equityWatch.some((row) => Number.isFinite(row.distanceToNearestStationKm)),
    [data.equityWatch]
  );

  const portfolioRows = useMemo<TableRow[]>(() => {
    const labels = ensureUniqueLabels(portfolioChoice.map((row) => row.locationLabel));
    return portfolioChoice.map((row, idx) => ({
      Location: labels[idx] ?? row.locationLabel,
      "Recommended action": row.action,
      "Estimated cost ($)": row.costUsd.toFixed(1),
      "Estimated prevented requests (30d)": row.preventedCalls30d.toFixed(1),
      "ROI: prevented requests per $1k": row.roiCallsPreventedPer1k.toFixed(1),
      "Underserved area": row.underservedFlag ? "True" : "False"
    }));
  }, [portfolioChoice]);

  const actionMix = useMemo(() => actionMixRows(portfolioChoice), [portfolioChoice]);
  const alertCounts = useMemo(() => alertLevelCounts(data.alerts), [data.alerts]);

  const planMarkdown = selectedRow
    ? actionPlanText(selectedRow.areaLabel, selectedRow, selectedChance, selectedBaseline30d, selectedAdjusted30d)
    : "";
  const parsedPitch = useMemo(() => parsePitchBrief(data.pitchBrief || ""), [data.pitchBrief]);

  const showAreaControl = activeTab === "overview" || activeTab === "innovation";
  const showTimelineControl = activeTab === "overview";
  const showBudgetControl = activeTab === "budget" && budgetOptions.length > 0;

  if (data.zoneBases.length === 0) {
    return (
      <div className="empty-shell">
        <h1>The Montgomery Guardian</h1>
        <p>No dashboard data found. Prepare artifacts in `data/served` (or `data/processed` for local dev).</p>
      </div>
    );
  }

  return (
    <div className="dash-shell">
      <aside className="dash-rail">
        <div className="rail-logo">TGM</div>
        <nav className="rail-nav" aria-label="Dashboard Tabs">
          {NAV_TABS.map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={activeTab === tab.key ? "rail-tab active" : "rail-tab"}
              onClick={() => setActiveTab(tab.key)}
            >
              <strong>{tab.label}</strong>
              <span>{tab.hint}</span>
            </button>
          ))}
        </nav>
      </aside>

      <main className="dash-main">
        <header className="dash-header">
          <div>
            <h1>The Montgomery Guardian</h1>
            <p>A simple city map of where service needs are likely to rise next.</p>
            {generatedDateOnly ? <span className="caption">Data last updated: {generatedDateOnly}</span> : null}
          </div>

          <div className="tab-indicator">{NAV_TABS.find((tab) => tab.key === activeTab)?.label}</div>
        </header>

        {showAreaControl || showTimelineControl || showBudgetControl ? (
          <section className="tab-controls">
            {showAreaControl ? (
              <label className="select-wrap inline">
                <span>Choose an area to inspect</span>
                <select value={selectedRow?.h3Cell ?? ""} onChange={(event) => setSelectedH3Cell(event.target.value)}>
                  {topCells.map((row) => (
                    <option key={row.h3Cell} value={row.h3Cell}>
                      {row.areaLabelDisplay}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            {showTimelineControl ? (
              <label className="slider-box">
                <span>Timeline: move to see how needs shift over the next 30 days</span>
                <input
                  type="range"
                  min={0}
                  max={30}
                  step={1}
                  value={timelineDay}
                  onChange={(event) => setTimelineDay(Number(event.target.value))}
                />
                <strong>
                  Day {timelineDay} ({selectedDateIso})
                </strong>
              </label>
            ) : null}
            {showBudgetControl ? (
              <label className="select-wrap inline budget-select">
                <span>Choose budget scenario</span>
                <select value={effectiveBudget} onChange={(event) => setBudgetChoice(Number(event.target.value))}>
                  {budgetOptions.map((option) => (
                    <option key={option} value={option}>
                      ${option.toLocaleString()}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
          </section>
        ) : null}

        {activeTab === "overview" ? (
          <section className="quick-guide card">
            <div className="quick-guide-head">
              <h2>How To Read This Dashboard In 60 Seconds</h2>
              <button type="button" className="quick-toggle" onClick={() => setShowQuickGuide((v) => !v)}>
                {showQuickGuide ? "Hide" : "Show"}
              </button>
            </div>
            {showQuickGuide ? (
              <div className="quick-guide-grid">
                <p>
                  <strong>1. Start in Overview:</strong> Move the timeline and scan the map to see where service demand is rising.
                </p>
                <p>
                  <strong>2. Open Area Snapshot:</strong> Pick a zone to see expected demand, uncertainty range, local code issues, and 311 history.
                </p>
                <p>
                  <strong>3. Use Innovation + Budget:</strong> Test interventions, then choose a budget scenario to see actions and expected prevented requests.
                </p>
                <p>
                  <strong>4. Finish in Operations + Alerts:</strong> Prioritize top areas, review uncertain zones, and monitor early-warning/equity signals.
                </p>
              </div>
            ) : null}
          </section>
        ) : null}

        {activeTab === "overview" ? (
          <>
            <section className="kpi-grid">
              <article className="card hero">
                <h3>Expected Requests This Month</h3>
                <p className="metric-large">{compact(totalPredicted30d)}</p>
                <span>Projected total service requests from all displayed areas.</span>
              </article>
              <article className="card">
                <h3>Areas Needing Attention Today</h3>
                <p className="metric-medium">{highRiskCount}</p>
              </article>
              <article className="card">
                <h3>Expected Requests On Selected Day</h3>
                <p className="metric-medium">{forecastDayTotal.toFixed(0)}</p>
              </article>
              <article className="card">
                <h3>Date</h3>
                <p className="metric-medium">{selectedDateIso}</p>
              </article>
            </section>

            <section className="callout-row">
              <p>
                For {selectedDateIso}, the model expects about {forecastDayTotal.toFixed(0)} citywide requests. Taller and warmer-colored
                areas on the map are places to prioritize.
              </p>
              <p>
                This is an estimate, not an exact count. "Expected requests in next 30 days" means projected total service requests from
                that location over the upcoming month.
              </p>
              <p>
                Forecast confidence band (citywide): +/- {(data.totalUncertainty30d / 2).toFixed(0)} requests ({data.uncertaintyPct.toFixed(1)}%
                uncertainty width).
              </p>
            </section>

            <section className="map-grid">
              <article className="card map-card">
                <RiskMap zones={mapZones} selectedH3Cell={selectedRow?.h3Cell} />
                <p className="caption">Color guide: blue = lower need, orange = higher need. Height also increases with need.</p>
                <p className="caption">Location names are based on the nearest known place in the city dataset.</p>
              </article>

              <article className="card snapshot-card">
                <h2>Area Snapshot</h2>
                {selectedRow ? (
                  <div className="snapshot-metrics">
                    <p>
                      <strong>Expected requests this month:</strong> {formatRequestCount(selectedRow.predictedCallsNext30d)}
                    </p>
                    <p>
                      <strong>Forecast range this month:</strong> {formatRequestCount(selectedRow.predictedCallsNext30dP10)} to{" "}
                      {formatRequestCount(selectedRow.predictedCallsNext30dP90)}
                    </p>
                    <p>
                      <strong>Chance of requests today:</strong> {selectedRow.chanceToday}
                    </p>
                    <p>
                      <strong>What that means:</strong> {requestsPhrase(selectedRow.riskToday)}
                    </p>
                    <p>
                      <strong>Uncertainty score (30d):</strong> {formatRequestCount(selectedRow.predictionUncertainty30d)}
                    </p>
                    <p>
                      <strong>Code issues reported:</strong> {selectedRow.codeViolationsCount.toFixed(0)}
                    </p>
                    <p>
                      <strong>311 requests (history):</strong> {selectedRow.service311Count.toFixed(0)}
                    </p>
                    <p>
                      <strong>Places/POIs in this area:</strong> {selectedRow.poiCount.toFixed(0)}
                    </p>
                    <p>
                      <strong>Distance to Station (km):</strong> {formatKm(selectedRow.distanceToNearestStationKm, false)}
                    </p>
                    <p>
                      <strong>Distance to Nearest POI (km):</strong> {formatKm(selectedRow.distanceToNearestPoiKm, true)}
                    </p>
                  </div>
                ) : null}
                {selectedRow ? <div className="info-box">{roiText(selectedRow)}</div> : null}
              </article>
            </section>
          </>
        ) : null}

        {activeTab === "innovation" ? (
          <section className="card">
            <h2>Innovation Lab</h2>
            <div className="slider-grid">
              <label>
                <span>Code cleanup effort ({codeCleanupPct}%)</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={codeCleanupPct}
                  onChange={(event) => setCodeCleanupPct(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Rapid response coverage ({rapidResponsePct}%)</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={rapidResponsePct}
                  onChange={(event) => setRapidResponsePct(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Hotspot patrol intensity ({hotspotPatrolPct}%)</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={5}
                  value={hotspotPatrolPct}
                  onChange={(event) => setHotspotPatrolPct(Number(event.target.value))}
                />
              </label>
              <label>
                <span>Special events pressure ({eventSurgePct}%)</span>
                <input
                  type="range"
                  min={0}
                  max={60}
                  step={5}
                  value={eventSurgePct}
                  onChange={(event) => setEventSurgePct(Number(event.target.value))}
                />
              </label>
            </div>

            <div className="kpi-grid small">
              <article className="card inset">
                <h3>Scenario Expected Requests (30d)</h3>
                <p className="metric-medium">{zones.reduce((sum, row) => sum + row.simulated30d, 0).toFixed(0)}</p>
              </article>
              <article className="card inset">
                <h3>Estimated Requests Prevented (30d)</h3>
                <p className="metric-medium">
                  {Math.max(0, zones.reduce((sum, row) => sum + row.predictedCallsNext30d, 0) - zones.reduce((sum, row) => sum + row.simulated30d, 0)).toFixed(0)}
                </p>
              </article>
              <article className="card inset">
                <h3>Scenario Pressure vs Baseline</h3>
                <p className="metric-medium">
                  {(
                    (zones.reduce((sum, row) => sum + row.simulated30d, 0) /
                      Math.max(zones.reduce((sum, row) => sum + row.predictedCallsNext30d, 0), 1e-9) -
                      1) *
                    100
                  ).toFixed(1)}
                  %
                </p>
              </article>
            </div>

            <div className="button-row">
              <button
                type="button"
                className="download-btn"
                onClick={() => downloadText("area_action_plan.md", planMarkdown, "text/markdown")}
                disabled={!selectedRow}
              >
                Download Area Action Plan (.md)
              </button>
            </div>

            <div className="chart-wrap">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={trendSeries} margin={{ top: 26, right: 34, left: 38, bottom: 28 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="dateLabel"
                    interval={4}
                    minTickGap={24}
                    tickMargin={8}
                    padding={{ left: 10, right: 22 }}
                    label={{
                      value: "Date (next 30 days)",
                      position: "bottom",
                      offset: 6,
                      fill: "#4e6286",
                      fontSize: 12
                    }}
                  />
                  <YAxis
                    width={82}
                    tickMargin={8}
                    tickFormatter={(value: number) => (value < 1 ? "<1" : value.toFixed(0))}
                    label={{
                      value: "Expected service requests (per day)",
                      angle: -90,
                      position: "left",
                      offset: 8,
                      fill: "#4e6286",
                      fontSize: 12
                    }}
                  />
                  <Tooltip
                    labelFormatter={(value, payload) => {
                      const row = payload?.[0]?.payload as { date?: string } | undefined;
                      return `Date: ${row?.date ?? String(value)}`;
                    }}
                    formatter={(value: number, name: string) => [formatRequestsWithContext(Number(value)), name]}
                  />
                  <Legend verticalAlign="top" height={28} />
                  <Line
                    type="monotone"
                    dataKey="baseline"
                    name="Baseline (current operations)"
                    stroke="#2563eb"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                  <Line
                    type="monotone"
                    dataKey="scenario"
                    name="Scenario (with selected actions)"
                    stroke="#f97316"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <p className="caption">
              Scenario Lab chart: each point is the expected number of service requests for that day in the selected area. Blue shows
              baseline operations; orange shows your scenario.
            </p>
          </section>
        ) : null}

        {activeTab === "budget" ? (
          <section className="card budget-card">
            <h2>Budget Optimizer (City Operations AI)</h2>
            {data.budgets.length > 0 && data.portfolio.length > 0 ? (
              <>
                <div className="budget-kpis">
                  <article className="card inset">
                    <h3>Planned Actions</h3>
                    <p className="metric-medium">{portfolioChoice.length}</p>
                  </article>
                  <article className="card inset">
                    <h3>Cells Covered</h3>
                    <p className="metric-medium">{new Set(portfolioChoice.map((row) => row.h3Cell)).size}</p>
                  </article>
                  <article className="card inset">
                    <h3>Prevented Requests (30d)</h3>
                    <p className="metric-medium">{portfolioChoice.reduce((sum, row) => sum + row.preventedCalls30d, 0).toFixed(0)}</p>
                  </article>
                  <article className="card inset">
                    <h3>Equity Spend Share</h3>
                    <p className="metric-medium">
                      {(
                        (portfolioChoice
                          .filter((row) => row.underservedFlag)
                          .reduce((sum, row) => sum + row.costUsd, 0) /
                          Math.max(
                            portfolioChoice.reduce((sum, row) => sum + row.costUsd, 0),
                            1e-9
                          )) *
                        100
                      ).toFixed(1)}
                      %
                    </p>
                  </article>
                </div>

                {budgetInfo ? (
                  <p className="caption budget-note">
                    Budget used: ${budgetInfo.spentUsd.toLocaleString(undefined, { maximumFractionDigits: 0 })} (
                    {budgetInfo.budgetUtilizationPct.toFixed(1)}% of budget). Avg ROI:{" "}
                    {`${budgetInfo.avgRoiCallsPreventedPer1k.toFixed(2)} prevented requests per $1k.`}
                  </p>
                ) : null}

                <div className="chart-wrap compact">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={actionMix} layout="vertical" margin={{ top: 10, right: 20, left: 30, bottom: 20 }}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <Legend verticalAlign="top" height={26} />
                      <XAxis
                        type="number"
                        tickFormatter={(value) => Number(value).toFixed(0)}
                        label={{
                          value: "Estimated prevented requests (30d)",
                          position: "insideBottom",
                          offset: -8,
                          fill: "#4e6286",
                          fontSize: 12
                        }}
                      />
                      <YAxis type="category" dataKey="action" width={170} tick={{ fontSize: 11 }} />
                      <Tooltip formatter={(value: number) => [Number(value).toFixed(1), "Prevented requests (30d)"]} />
                      <Bar dataKey="prevented" name="Prevented requests (30d)" fill="#fb7185" radius={[0, 8, 8, 0]}>
                        <LabelList dataKey="prevented" position="right" formatter={(value: number) => Number(value).toFixed(1)} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <p className="caption">Each bar shows expected prevented requests over the next 30 days by intervention type.</p>

                <DataTable
                  columns={[
                    "Location",
                    "Recommended action",
                    "Estimated cost ($)",
                    "Estimated prevented requests (30d)",
                    "ROI: prevented requests per $1k",
                    "Underserved area"
                  ]}
                  rows={portfolioRows}
                />
              </>
            ) : (
              <p className="caption">Run the optimizer step to unlock budget planning: intervention portfolio, alerts, and equity watchlist.</p>
            )}
          </section>
        ) : null}

        {activeTab === "operations" ? (
          <>
            <section className="card">
              <h2>What This Means Today</h2>
              <div className="triple-info">
                <div>Start with the top 3 areas below for cleanup and enforcement.</div>
                <div>Prioritize places with high code issues and longer distance to a station.</div>
                <div>Check this view weekly and adjust the priority list.</div>
              </div>
            </section>

            <section className="card">
              <h2>Top Areas Today</h2>
              <div className="stack-grid">
                {topAreas.map((row) => (
                  <article key={row.area} className="stack-card">
                    <h3>{row.area}</h3>
                    <p>
                      <strong>Recommended action:</strong> {row.recommendedAction}
                    </p>
                    <p>
                      <strong>Chance of requests today:</strong> {row.chance}
                    </p>
                    <p>
                      <strong>Expected requests in next 30 days:</strong> {row.expected30d}
                    </p>
                    <p>
                      <strong>Open neighborhood code complaints:</strong> {row.codeComplaints}
                    </p>
                    <p>
                      <strong>Past city service requests (311):</strong> {row.service311}
                    </p>
                    <p>
                      <strong>Blind-spot signal:</strong> {row.blindSpot}
                    </p>
                    <p>
                      <strong>Why this area is prioritized:</strong> {row.why}
                    </p>
                  </article>
                ))}
              </div>
              <p className="caption">
                Column meanings: Open neighborhood code complaints = unresolved property/neighborhood issues. Past city service requests
                (311) = historical requests from residents in that location.
              </p>
            </section>

            <section className="card">
              <h2>Intervention Leaderboard</h2>
              <div className="leaderboard-list">
                {leaderboard.map((row) => (
                  <article key={row.location} className="leaderboard-item">
                    <div className="leaderboard-head">
                      <h3>{row.location}</h3>
                      <span>{row.preventable30d} preventable (30d)</span>
                    </div>
                    <div className="leaderboard-bar">
                      <span style={{ width: `${row.barPct}%` }} />
                    </div>
                    <div className="leaderboard-meta">
                      <p>Baseline: {row.baseline30d}</p>
                      <p>Scenario: {row.scenario30d}</p>
                      <p>Blind-spot: {row.blindSpot}</p>
                    </div>
                  </article>
                ))}
              </div>
              <p className="caption">
                Use this leaderboard to prioritize where combined cleanup + response actions can create the largest impact.
              </p>
            </section>

            <section className="card">
              <h2>AI Review Queue (High Uncertainty Zones)</h2>
              <div className="review-grid">
                {reviewQueue.map((row) => (
                  <article key={row.location} className="stack-card review-card">
                    <h3>{row.location}</h3>
                    <div className="review-metrics">
                      <p>
                        <strong>Forecast spread (30d):</strong> {row.spread30d}
                      </p>
                      <p>
                        <strong>Lower estimate (30d):</strong> {row.lower30d}
                      </p>
                      <p>
                        <strong>Upper estimate (30d):</strong> {row.upper30d}
                      </p>
                      <p>
                        <strong>Open code complaints:</strong> {row.codeComplaints}
                      </p>
                      <p>
                        <strong>Past 311 requests:</strong> {row.service311}
                      </p>
                    </div>
                    <p>
                      <strong>Action:</strong> {row.action}
                    </p>
                  </article>
                ))}
              </div>
            </section>
          </>
        ) : null}

        {activeTab === "alerts" ? (
          <section className="split-grid">
            <article className="card">
              <h2>Early Warning Radar</h2>
              {data.alerts.length > 0 ? (
                <>
                  <div className="chart-wrap compact">
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={alertCounts} margin={{ top: 10, right: 20, left: 20, bottom: 26 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis
                          dataKey="level"
                          label={{
                            value: "Alert level",
                            position: "insideBottom",
                            offset: -10,
                            fill: "#4e6286",
                            fontSize: 12
                          }}
                        />
                        <YAxis
                          allowDecimals={false}
                          label={{
                            value: "Number of zones",
                            angle: -90,
                            position: "insideLeft",
                            fill: "#4e6286",
                            fontSize: 12
                          }}
                        />
                        <Tooltip
                          formatter={(value: number) => [Number(value).toFixed(0), "Zones"]}
                          labelFormatter={(label: string) => `Alert level: ${label}`}
                        />
                        <Bar dataKey="count" name="Zones by alert level" fill="#fb7185" radius={8}>
                          <LabelList dataKey="count" position="insideTop" offset={10} fill="#ffffff" />
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="caption">Bars show how many zones are currently flagged at each alert level.</p>

                  <div className="stack-grid compact">
                    {alerts.map((row) => (
                      <article key={`${row.location}-${row.level}-${row.stress}`} className="stack-card">
                        <h3>{row.location}</h3>
                        <p>
                          <strong>Alert level:</strong> {row.level}
                        </p>
                        <p>
                          <strong>Why flagged:</strong> {row.reason}
                        </p>
                        <p>
                          <strong>Expected requests (30d):</strong> {row.expected30d}
                        </p>
                        <p>
                          <strong>High-end estimate (30d):</strong> {row.highEnd30d}
                        </p>
                        <p>
                          <strong>Stress score:</strong> {row.stress}
                        </p>
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <p className="caption">Early warning alerts will appear once `early_warning_alerts.csv` is available.</p>
              )}
            </article>

            <article className="card">
              <h2>Equity Watchlist</h2>
              {data.equityWatch.length > 0 ? (
                <>
                  {!hasEquityStationDistance ? (
                    <p className="caption">Station distance is hidden here because station-location data is not available.</p>
                  ) : null}
                  <div className="stack-grid equity-grid">
                    {equity.map((row) => (
                      <article key={`${row.location}-${row.risk}-${row.stress}`} className="stack-card">
                        <h3>{row.location}</h3>
                        <p>
                          <strong>Risk level:</strong> {row.risk}
                        </p>
                        {hasEquityStationDistance ? (
                          <p>
                            <strong>Distance to station (km):</strong> {row.stationKm}
                          </p>
                        ) : null}
                        <p>
                          <strong>Stress score:</strong> {row.stress}
                        </p>
                        <p>
                          <strong>Primary concern:</strong> {row.concern}
                        </p>
                      </article>
                    ))}
                  </div>
                </>
              ) : (
                <p className="caption">Equity watchlist will appear once `equity_watchlist.csv` is available.</p>
              )}
            </article>
          </section>
        ) : null}

        {activeTab === "pitch" ? (
          <section className="card">
            <h2>Pitch Brief Generator</h2>
            <div className="button-row">
              <button type="button" className="download-btn" onClick={() => setPitchVisible(true)}>
                Generate Pitch Narrative
              </button>
              <button
                type="button"
                className="download-btn secondary"
                onClick={() => downloadText("pitch_summary.md", data.pitchBrief || "", "text/markdown")}
                disabled={!data.pitchBrief}
              >
                Download Brief (.md)
              </button>
            </div>
            {pitchVisible ? (
              data.pitchBrief ? (
                <div className="pitch-render">
                  <article className="pitch-hero-card">
                    <h3>{parsedPitch.title}</h3>
                    <p>Ready-to-present narrative generated from current model outputs and city datasets.</p>
                    {parsedPitch.lead.length > 0 ? (
                      <ul>
                        {parsedPitch.lead.slice(0, 4).map((line) => (
                          <li key={line}>{line}</li>
                        ))}
                      </ul>
                    ) : null}
                  </article>

                  <div className="pitch-section-grid">
                    {parsedPitch.sections.map((section) => (
                      <article key={section.title} className="pitch-section-card">
                        <h3>{section.title}</h3>
                        {section.paragraphs.map((line) => (
                          <p key={`${section.title}-${line}`}>{line}</p>
                        ))}
                        {section.bullets.length > 0 ? (
                          <ul>
                            {section.bullets.map((line) => (
                              <li key={`${section.title}-${line}`}>{line}</li>
                            ))}
                          </ul>
                        ) : null}
                      </article>
                    ))}
                  </div>

                  <details className="pitch-raw">
                    <summary>Show raw markdown brief</summary>
                    <pre className="pitch-box">{data.pitchBrief}</pre>
                  </details>
                </div>
              ) : (
                <pre className="pitch-box">No pitch_summary.md found in data/served.</pre>
              )
            ) : null}
            <div className="caption">Model metrics reference: R2 {valueText(data.r2, 4)}, Moran&apos;s I {valueText(data.moranI, 6)}.</div>
          </section>
        ) : null}
      </main>
    </div>
  );
}
