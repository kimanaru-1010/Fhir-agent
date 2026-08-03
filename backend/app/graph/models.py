"""Domain models for Healthcare — auto-generated from ontology."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

class Person(BaseModel):
    """Entity model for Person."""

    name: str = Field(...)
    email: str | None = None
    role: str | None = None
    description: str | None = None

class Organization(BaseModel):
    """Entity model for Organization."""

    name: str = Field(...)
    description: str | None = None
    industry: str | None = None

class Location(BaseModel):
    """Entity model for Location."""

    name: str = Field(...)
    address: str | None = None
    latitude: float | None = None
    longitude: float | None = None

class Event(BaseModel):
    """Entity model for Event."""

    name: str = Field(...)
    date: datetime | None = None
    description: str | None = None

class Object(BaseModel):
    """Entity model for Object."""

    name: str = Field(...)
    description: str | None = None

class PatientBloodTypeEnum(str, Enum):
    A_PLUS = "A+"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B_MINUS = "B-"
    AB_PLUS = "AB+"
    AB_MINUS = "AB-"
    O_PLUS = "O+"
    O_MINUS = "O-"

class Patient(BaseModel):
    """Entity model for Patient."""

    patient_id: str = Field(...)
    name: str = Field(...)
    date_of_birth: date | None = None
    blood_type: PatientBloodTypeEnum | None = None
    allergies: str | None = None

class ProviderSpecialtyEnum(str, Enum):
    GENERAL_PRACTICE = "general_practice"
    CARDIOLOGY = "cardiology"
    ONCOLOGY = "oncology"
    NEUROLOGY = "neurology"
    ORTHOPEDICS = "orthopedics"
    PEDIATRICS = "pediatrics"
    RADIOLOGY = "radiology"
    SURGERY = "surgery"
    EMERGENCY = "emergency"
    PSYCHIATRY = "psychiatry"

class Provider(BaseModel):
    """Entity model for Provider."""

    provider_id: str = Field(...)
    name: str = Field(...)
    specialty: ProviderSpecialtyEnum | None = None
    license_number: str | None = None

class DiagnosisCategoryEnum(str, Enum):
    CHRONIC = "chronic"
    ACUTE = "acute"
    INFECTIOUS = "infectious"
    AUTOIMMUNE = "autoimmune"
    GENETIC = "genetic"
    MENTAL_HEALTH = "mental_health"

class DiagnosisSeverityEnum(str, Enum):
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"

class Diagnosis(BaseModel):
    """Entity model for Diagnosis."""

    icd_code: str = Field(...)
    name: str = Field(...)
    category: DiagnosisCategoryEnum | None = None
    severity: DiagnosisSeverityEnum | None = None

class TreatmentTreatmentTypeEnum(str, Enum):
    MEDICATION = "medication"
    SURGERY = "surgery"
    THERAPY = "therapy"
    PROCEDURE = "procedure"
    IMAGING = "imaging"
    LAB_TEST = "lab_test"

class TreatmentStatusEnum(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class Treatment(BaseModel):
    """Entity model for Treatment."""

    treatment_id: str = Field(...)
    name: str = Field(...)
    treatment_type: TreatmentTreatmentTypeEnum | None = None
    status: TreatmentStatusEnum | None = None
    start_date: datetime | None = None
    end_date: datetime | None = None

class EncounterEncounterTypeEnum(str, Enum):
    INPATIENT = "inpatient"
    OUTPATIENT = "outpatient"
    EMERGENCY = "emergency"
    TELEHEALTH = "telehealth"

class Encounter(BaseModel):
    """Entity model for Encounter."""

    encounter_id: str = Field(...)
    encounter_type: EncounterEncounterTypeEnum | None = None
    date: datetime = Field(...)
    chief_complaint: str | None = None
    disposition: str | None = None

class FacilityFacilityTypeEnum(str, Enum):
    HOSPITAL = "hospital"
    CLINIC = "clinic"
    URGENT_CARE = "urgent_care"
    LAB = "lab"
    PHARMACY = "pharmacy"
    REHAB_CENTER = "rehab_center"

class Facility(BaseModel):
    """Entity model for Facility."""

    facility_id: str = Field(...)
    name: str = Field(...)
    facility_type: FacilityFacilityTypeEnum | None = None
    bed_count: int | None = None

class Medication(BaseModel):
    """Entity model for Medication."""

    ndc_code: str = Field(...)
    name: str = Field(...)
    drug_class: str | None = None
    dosage_form: str | None = None
    contraindications: str | None = None

