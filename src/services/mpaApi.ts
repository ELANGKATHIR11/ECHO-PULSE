import { API_BASE_URL } from './api';

export interface MpaZone {
  id: string;
  name: string;
  state: string;
  sea_sector: string;
  established_year: number;
  area_sq_km: number;
  center_coords: [number, number]; // [lat, lng]
  boundary_polygon: [number, number][]; // [[lat, lng], ...]
  ecosystem_type: string;
  primary_protection_target: string;
  certifying_agency: string;
  threat_status: string;
  active_surveys_count: number;
  tagged_debris_count: number;
}

export interface MpaDebrisGeoTag {
  id: string;
  official_agency_ref: string;
  certifying_agency: string;
  mpa_id: string;
  mpa_name: string;
  sea_sector: string;
  target_class: 'PLASTIC' | 'METAL_SCRAP' | 'ELECTRICAL' | 'ELECTRONIC' | 'HUMAN';
  sub_category: string;
  marine_label: string;
  latitude: number;
  longitude: number;
  coordinates_dms: string;
  depth_meters: number;
  slant_range_meters: number;
  estimated_height_meters: number;
  target_dimensions_meters: {
    length: number;
    width: number;
    height: number;
  };
  threat_level: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  clean_coast_index_score: number;
  acoustic_snr_db: number;
  detection_confidence: number;
  survey_vessel: string;
  sonar_frequency_khz: number;
  verification_status: 'VERIFIED' | 'RECLAMATION_SCHEDULED' | 'RECLAIMED';
  tag_timestamp: string;
  notes: string;
}

export interface MpaSummaryMetrics {
  total_mpas: number;
  total_tagged_debris: number;
  critical_threat_count: number;
  total_mpa_area_sq_km: number;
  category_distribution: Record<string, number>;
  certifying_agencies: Record<string, number>;
  governing_frameworks: string[];
}

export const mpaApi = {
  async getMpaZones(): Promise<MpaZone[]> {
    try {
      const res = await fetch(`${API_BASE_URL}/gis/mpa-zones`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      return data.zones || [];
    } catch (e) {
      console.warn('Using local fallback for MPA zones:', e);
      return [
        {
          id: "MPA-IND-001",
          name: "Gulf of Mannar Marine National Park",
          state: "Tamil Nadu",
          sea_sector: "Gulf of Mannar / Palk Strait",
          established_year: 1986,
          area_sq_km: 560.0,
          center_coords: [9.1524, 79.2819],
          boundary_polygon: [
            [8.78, 78.15],
            [9.02, 78.50],
            [9.28, 79.15],
            [9.32, 79.45],
            [9.15, 79.52],
            [8.85, 78.95],
            [8.65, 78.35],
            [8.78, 78.15]
          ],
          ecosystem_type: "Coral Reef & Seagrass Meadows",
          primary_protection_target: "Dugong (Sea Cow), Coral Reefs, Olive Ridley Turtles",
          certifying_agency: "MoEFCC & NCCR (MoES)",
          threat_status: "CRITICAL (High ALDFG / Ghost Net Density)",
          active_surveys_count: 18,
          tagged_debris_count: 42
        },
        {
          id: "MPA-IND-002",
          name: "Marine National Park & Sanctuary, Gulf of Kutch",
          state: "Gujarat",
          sea_sector: "Arabian Sea (Gulf of Kutch)",
          established_year: 1982,
          area_sq_km: 457.9,
          center_coords: [22.4680, 69.5840],
          boundary_polygon: [
            [22.35, 69.15],
            [22.65, 69.45],
            [22.80, 70.10],
            [22.60, 70.25],
            [22.40, 69.85],
            [22.35, 69.15]
          ],
          ecosystem_type: "Mangroves, Corals & Mudflats",
          primary_protection_target: "Hard Corals, Sponges, Green Sea Turtles",
          certifying_agency: "Gujarat Forest Dept & CSIR-NIO",
          threat_status: "HIGH (Industrial Polymer & Maritime Shipping Corridor)",
          active_surveys_count: 14,
          tagged_debris_count: 28
        },
        {
          id: "MPA-IND-003",
          name: "Gahirmatha Marine Wildlife Sanctuary",
          state: "Odisha",
          sea_sector: "Bay of Bengal (Northern Corridor)",
          established_year: 1997,
          area_sq_km: 1435.0,
          center_coords: [20.7180, 86.9520],
          boundary_polygon: [
            [20.45, 86.75],
            [20.90, 86.85],
            [20.95, 87.15],
            [20.50, 87.05],
            [20.45, 86.75]
          ],
          ecosystem_type: "Open Coastal Waters & Estuary",
          primary_protection_target: "World's Largest Mass Nesting Site for Olive Ridley Turtles (Arribada)",
          certifying_agency: "Odisha Wildlife & NCCR (MoES)",
          threat_status: "CRITICAL (Monofilament Gillnet Entanglement Risk)",
          active_surveys_count: 22,
          tagged_debris_count: 35
        },
        {
          id: "MPA-IND-005",
          name: "Mahatma Gandhi Marine National Park (Wandoor)",
          state: "Andaman & Nicobar Islands",
          sea_sector: "Andaman Sea (South Andaman)",
          established_year: 1983,
          area_sq_km: 281.5,
          center_coords: [11.5830, 92.5920],
          boundary_polygon: [
            [11.45, 92.48],
            [11.72, 92.52],
            [11.70, 92.68],
            [11.42, 92.65],
            [11.45, 92.48]
          ],
          ecosystem_type: "Pristine Deep Ocean Coral Reefs & Mangrove Creeks",
          primary_protection_target: "50+ Coral Genera, Hawksbill Turtles, Saltwater Crocodiles",
          certifying_agency: "A&N Forest Dept & INCOIS",
          threat_status: "HIGH (Transboundary Pelagic Drift Plastics & Ghost Nets)",
          active_surveys_count: 16,
          tagged_debris_count: 24
        },
        {
          id: "MPA-IND-009",
          name: "Netrani Island Marine Biodiversity Zone",
          state: "Karnataka",
          sea_sector: "Arabian Sea (Central Coast)",
          established_year: 2012,
          area_sq_km: 15.4,
          center_coords: [14.0180, 74.3310],
          boundary_polygon: [
            [13.98, 74.29],
            [14.05, 74.31],
            [14.04, 74.37],
            [13.97, 74.35],
            [13.98, 74.29]
          ],
          ecosystem_type: "Pigeon Island Coral Pinnacle",
          primary_protection_target: "Blacktip Reef Sharks, Great Barracudas, Soft Coral Gardens",
          certifying_agency: "Karnataka Biodiversity Board & CMFRI",
          threat_status: "CRITICAL (High Density of Lost Trawl Nets on Pinnacles)",
          active_surveys_count: 8,
          tagged_debris_count: 17
        }
      ];
    }
  },

  async getDebrisGeoTags(params?: {
    mpa_id?: string;
    threat_level?: string;
    target_class?: string;
    certifying_agency?: string;
  }): Promise<MpaDebrisGeoTag[]> {
    try {
      const query = new URLSearchParams();
      if (params?.mpa_id) query.append('mpa_id', params.mpa_id);
      if (params?.threat_level) query.append('threat_level', params.threat_level);
      if (params?.target_class) query.append('target_class', params.target_class);
      if (params?.certifying_agency) query.append('certifying_agency', params.certifying_agency);

      const res = await fetch(`${API_BASE_URL}/gis/mpa-debris?${query.toString()}`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      return data.tags || [];
    } catch (e) {
      console.warn('Using local fallback for MPA debris tags:', e);
      return [
        {
          id: "TAG-GOM-2026-001",
          official_agency_ref: "NCCR-ML-2026-GOM-042",
          certifying_agency: "NCCR (MoES)",
          mpa_id: "MPA-IND-001",
          mpa_name: "Gulf of Mannar Marine National Park",
          sea_sector: "Gulf of Mannar",
          target_class: "PLASTIC",
          sub_category: "Derelict Ghost Gear & Synthetic Fishing Net",
          marine_label: "High-Density Polyethylene Synthetic Net Ensnared on Acropora Coral Reef",
          latitude: 9.1524,
          longitude: 79.2819,
          coordinates_dms: "09°09'08.6\"N, 79°16'54.8\"E",
          depth_meters: 32.4,
          slant_range_meters: 48.2,
          estimated_height_meters: 1.45,
          target_dimensions_meters: { length: 14.2, width: 8.6, height: 1.45 },
          threat_level: "CRITICAL",
          clean_coast_index_score: 14.8,
          acoustic_snr_db: 24.2,
          detection_confidence: 0.942,
          survey_vessel: "RV Sagar Nidhi (AUV DeepScan-4)",
          sonar_frequency_khz: 455.0,
          verification_status: "VERIFIED",
          tag_timestamp: "2026-08-22T10:45:00Z",
          notes: "Direct entanglement on critical seagrass corridor. Immediate AUV robotic grapple extraction recommended by NCCR Marine Litter survey team."
        },
        {
          id: "TAG-GOM-2026-002",
          official_agency_ref: "CMFRI-GOM-2026-118",
          certifying_agency: "CMFRI (ICAR)",
          mpa_id: "MPA-IND-001",
          mpa_name: "Gulf of Mannar Marine National Park",
          sea_sector: "Gulf of Mannar",
          target_class: "METAL_SCRAP",
          sub_category: "Shipwreck Hull Fragment & Structural Steel",
          marine_label: "Derelict Trawler Steel Keel Scrap on Benthic Reef Margin",
          latitude: 9.1845,
          longitude: 79.3102,
          coordinates_dms: "09°11'04.2\"N, 79°18'36.7\"E",
          depth_meters: 28.6,
          slant_range_meters: 35.8,
          estimated_height_meters: 2.10,
          target_dimensions_meters: { length: 9.4, width: 3.8, height: 2.1 },
          threat_level: "HIGH",
          clean_coast_index_score: 12.2,
          acoustic_snr_db: 28.6,
          detection_confidence: 0.918,
          survey_vessel: "FV Sagar Kanya",
          sonar_frequency_khz: 900.0,
          verification_status: "RECLAMATION_SCHEDULED",
          tag_timestamp: "2026-08-20T14:15:00Z",
          notes: "Corroded ferrous scrap producing heavy acoustic multipath reflections. Certified as vessel wreckage fragment by CMFRI Mandapam Regional Centre."
        },
        {
          id: "TAG-GAH-2026-001",
          official_agency_ref: "NCCR-ODS-2026-014",
          certifying_agency: "NCCR (MoES)",
          mpa_id: "MPA-IND-003",
          mpa_name: "Gahirmatha Marine Wildlife Sanctuary",
          sea_sector: "Bay of Bengal",
          target_class: "PLASTIC",
          sub_category: "Monofilament Gillnet Obstruction",
          marine_label: "Entangled High-Tenacity Monofilament Gillnet Cluster",
          latitude: 20.7240,
          longitude: 86.9610,
          coordinates_dms: "20°43'26.4\"N, 86°57'39.6\"E",
          depth_meters: 24.2,
          slant_range_meters: 54.0,
          estimated_height_meters: 1.80,
          target_dimensions_meters: { length: 22.0, width: 6.5, height: 1.8 },
          threat_level: "CRITICAL",
          clean_coast_index_score: 18.2,
          acoustic_snr_db: 26.4,
          detection_confidence: 0.965,
          survey_vessel: "ICGS Amrit Kaur (Patrol Unit)",
          sonar_frequency_khz: 455.0,
          verification_status: "VERIFIED",
          tag_timestamp: "2026-08-19T06:20:00Z",
          notes: "Direct obstruction along active Olive Ridley breeding migration fairway. Immediate Indian Coast Guard clearance authorized."
        },
        {
          id: "TAG-KUT-2026-001",
          official_agency_ref: "NIO-GUJ-2026-088",
          certifying_agency: "CSIR-NIO",
          mpa_id: "MPA-IND-002",
          mpa_name: "Marine National Park & Sanctuary, Gulf of Kutch",
          sea_sector: "Arabian Sea",
          target_class: "PLASTIC",
          sub_category: "Industrial Polypropylene Packaging & Trawl Debris",
          marine_label: "Dense Polymer Marine Litter & Discarded Commercial Trawl Netting",
          latitude: 22.4810,
          longitude: 69.6120,
          coordinates_dms: "22°28'51.6\"N, 69°36'43.2\"E",
          depth_meters: 18.4,
          slant_range_meters: 32.1,
          estimated_height_meters: 1.20,
          target_dimensions_meters: { length: 11.5, width: 7.0, height: 1.2 },
          threat_level: "HIGH",
          clean_coast_index_score: 15.4,
          acoustic_snr_db: 23.5,
          detection_confidence: 0.892,
          survey_vessel: "RV Sindhu Sankalp",
          sonar_frequency_khz: 455.0,
          verification_status: "VERIFIED",
          tag_timestamp: "2026-08-21T09:10:00Z",
          notes: "Snagged across Pirotan Island coral shoal. Clean Coast Index indicates heavy industrial shipping runoff."
        },
        {
          id: "TAG-AND-2026-001",
          official_agency_ref: "INCOIS-AN-2026-033",
          certifying_agency: "INCOIS (MoES)",
          mpa_id: "MPA-IND-005",
          mpa_name: "Mahatma Gandhi Marine National Park (Wandoor)",
          sea_sector: "Andaman Sea",
          target_class: "PLASTIC",
          sub_category: "Transboundary Pelagic Ghost Net",
          marine_label: "Deep Ocean Drifting Monofilament Tangle on Coral Seamount",
          latitude: 11.5940,
          longitude: 92.6080,
          coordinates_dms: "11°35'38.4\"N, 92°36'28.8\"E",
          depth_meters: 46.2,
          slant_range_meters: 62.4,
          estimated_height_meters: 2.40,
          target_dimensions_meters: { length: 28.5, width: 12.0, height: 2.4 },
          threat_level: "CRITICAL",
          clean_coast_index_score: 17.1,
          acoustic_snr_db: 25.1,
          detection_confidence: 0.948,
          survey_vessel: "RV Sagar Kanya (SAS High-Res)",
          sonar_frequency_khz: 900.0,
          verification_status: "VERIFIED",
          tag_timestamp: "2026-08-17T15:45:00Z",
          notes: "Large ghost net trapped on deep coral pinnacle in Rutland Passage. Verified via INCOIS oceanic drift modeling trajectory."
        },
        {
          id: "TAG-NET-2026-001",
          official_agency_ref: "CMFRI-KAR-2026-056",
          certifying_agency: "CMFRI (ICAR)",
          mpa_id: "MPA-IND-009",
          mpa_name: "Netrani Island Marine Biodiversity Zone",
          sea_sector: "Arabian Sea",
          target_class: "PLASTIC",
          sub_category: "Heavy Commercial Trawl Webbing",
          marine_label: "High-Strength Synthetic Trawl Webbing Ensnared on Rocky Pinnacle",
          latitude: 14.0210,
          longitude: 74.3350,
          coordinates_dms: "14°01'15.6\"N, 74°20'06.0\"E",
          depth_meters: 26.5,
          slant_range_meters: 38.0,
          estimated_height_meters: 1.60,
          target_dimensions_meters: { length: 16.0, width: 9.2, height: 1.6 },
          threat_level: "CRITICAL",
          clean_coast_index_score: 16.2,
          acoustic_snr_db: 27.0,
          detection_confidence: 0.931,
          survey_vessel: "CMFRI Research Vessel Unit",
          sonar_frequency_khz: 455.0,
          verification_status: "VERIFIED",
          tag_timestamp: "2026-08-16T12:00:00Z",
          notes: "Severe entanglement hazard for scuba divers and pelagic fish schools. Flagged by CMFRI Karwar research station."
        }
      ];
    }
  },

  async getIndianEez(): Promise<[number, number][]> {
    try {
      const res = await fetch(`${API_BASE_URL}/gis/indian-eez`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      return data.coordinates || [];
    } catch (e) {
      console.warn('Using local fallback for EEZ boundary:', e);
      return [
        [23.5, 68.0],
        [20.5, 67.2],
        [17.0, 68.5],
        [13.0, 70.5],
        [9.0, 71.8],
        [6.5, 75.5],
        [5.8, 79.5],
        [8.0, 82.0],
        [10.5, 84.5],
        [14.5, 85.8],
        [18.0, 88.2],
        [21.2, 89.5],
        [21.5, 87.5],
        [19.0, 85.0],
        [15.5, 81.0],
        [12.0, 80.5],
        [8.5, 78.0],
        [11.5, 75.0],
        [16.0, 73.0],
        [20.0, 72.0],
        [23.5, 68.0]
      ];
    }
  },

  async getMpaSummary(): Promise<MpaSummaryMetrics | null> {
    try {
      const res = await fetch(`${API_BASE_URL}/gis/mpa-summary`);
      if (!res.ok) throw new Error(`HTTP error ${res.status}`);
      const data = await res.json();
      return data.metrics || null;
    } catch (e) {
      console.warn('Using local fallback for MPA summary:', e);
      return {
        total_mpas: 10,
        total_tagged_debris: 8,
        critical_threat_count: 5,
        total_mpa_area_sq_km: 28965.0,
        category_distribution: {
          PLASTIC: 5,
          METAL_SCRAP: 2,
          ELECTRICAL: 1
        },
        certifying_agencies: {
          "NCCR (MoES)": 3,
          "CMFRI (ICAR)": 2,
          "CSIR-NIO": 2,
          "INCOIS (MoES)": 1
        },
        governing_frameworks: [
          "Swachh Sagar Surakshit Sagar (MoES / NCCR)",
          "Wildlife Protection Act 1972 (Schedule I Marine Habitats)",
          "Coastal Regulation Zone (CRZ-I Ecologically Sensitive Areas)",
          "United Nations SDG-14 (Life Below Water / ALDFG Mitigation)"
        ]
      };
    }
  }
};

