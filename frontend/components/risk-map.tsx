"use client";

import { useMemo } from "react";
import { MapContainer, Polygon, Popup, TileLayer } from "react-leaflet";
import * as h3 from "h3-js";

export type MapZone = {
  h3Cell: string;
  areaLabel: string;
  centroidLatitude: number;
  centroidLongitude: number;
  predictedCallsNext30d: number;
  predictedCallsNext30dP10: number;
  predictedCallsNext30dP90: number;
  predictionUncertainty30d: number;
  codeViolationsCount: number;
  service311Count: number;
  riskToday: number;
  forecastTodayCount: string;
  forecastTodayText: string;
  chanceToday: string;
  distanceToNearestStationKm: number;
  distanceToNearestPoiKm: number;
  color: [number, number, number, number];
};

type Props = {
  zones: MapZone[];
  selectedH3Cell?: string;
};

type PolygonFeature = {
  zone: MapZone;
  positions: [number, number][];
};

function fillColor(color: [number, number, number, number]): string {
  return `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${(color[3] / 255).toFixed(3)})`;
}

function safeKm(value: number, allowZero = true): string {
  if (!Number.isFinite(value)) return "Not available";
  if (!allowZero && value <= 0) return "Not available";
  return value.toFixed(2);
}

export default function RiskMap({ zones, selectedH3Cell }: Props) {
  const features = useMemo<PolygonFeature[]>(() => {
    const out: PolygonFeature[] = [];
    for (const zone of zones) {
      try {
        const boundary = h3.cellToBoundary(zone.h3Cell, true) as [number, number][];
        const positions = boundary.map(([lng, lat]) => [lat, lng] as [number, number]);
        if (positions.length > 0) {
          out.push({ zone, positions });
        }
      } catch {
        continue;
      }
    }
    return out;
  }, [zones]);

  const center = useMemo<[number, number]>(() => {
    const selected = zones.find((z) => z.h3Cell === selectedH3Cell);
    if (selected && Number.isFinite(selected.centroidLatitude) && Number.isFinite(selected.centroidLongitude)) {
      return [selected.centroidLatitude, selected.centroidLongitude];
    }
    if (zones.length > 0) {
      const lat = zones.reduce((sum, zone) => sum + zone.centroidLatitude, 0) / zones.length;
      const lon = zones.reduce((sum, zone) => sum + zone.centroidLongitude, 0) / zones.length;
      if (Number.isFinite(lat) && Number.isFinite(lon)) {
        return [lat, lon];
      }
    }
    return [32.3668, -86.3];
  }, [selectedH3Cell, zones]);

  return (
    <div className="map-wrap">
      <MapContainer center={center} zoom={10.8} scrollWheelZoom style={{ width: "100%", height: "100%" }}>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; CARTO'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />
        {features.map(({ zone, positions }) => {
          const selected = zone.h3Cell === selectedH3Cell;
          return (
            <Polygon
              key={zone.h3Cell}
              positions={positions}
              pathOptions={{
                fillColor: fillColor(zone.color),
                color: selected ? "#f8fafc" : "rgba(15, 23, 42, 0.35)",
                weight: selected ? 2 : 0.6,
                fillOpacity: selected ? 0.92 : 0.82
              }}
            >
              <Popup>
                <strong>{zone.areaLabel}</strong>
                <br />
                <strong>Expected requests today:</strong> {zone.forecastTodayCount}
                <br />
                <strong>Meaning:</strong> {zone.forecastTodayText}
                <br />
                <strong>Chance today:</strong> {zone.chanceToday}
                <br />
                <strong>Expected requests (30d):</strong> {Math.round(zone.predictedCallsNext30d)}
                <br />
                <strong>Forecast range (30d):</strong> {Math.round(zone.predictedCallsNext30dP10)} to {Math.round(zone.predictedCallsNext30dP90)}
                <br />
                <strong>Code issues:</strong> {Math.round(zone.codeViolationsCount)}
                <br />
                <strong>311 requests:</strong> {Math.round(zone.service311Count)}
                <br />
                <strong>Distance to station (km):</strong> {safeKm(zone.distanceToNearestStationKm, false)}
                <br />
                <strong>Distance to nearest POI (km):</strong> {safeKm(zone.distanceToNearestPoiKm, true)}
              </Popup>
            </Polygon>
          );
        })}
      </MapContainer>
    </div>
  );
}
