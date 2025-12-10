# PsoMetric Backend - Class Diagram

## Overview
This document illustrates the class structure and relationships within the PsoMetric Backend system. The system is built using FastAPI and follows a modular architecture separating services, schemas, database interactions, and API routers.

## System Class Diagram

```mermaid
classDiagram
    %% Core AI Service
    class AIEngine {
        -sniper: YOLO
        -judge: EfficientNet
        -device: torch.device
        -judge_transform: transforms.Compose
        +__init__()
        +load_models()
        +analyze_image(original_image: Image) : dict
        +calculate_lesion_metrics(pil_crop: Image, lesion_mask: np.array) : dict
        -image_to_base64(numpy_image: np.array) : str
        -white_balance(cv_img: np.array) : np.array
        -_prepare_image_data(cv_img, cv_img_original, lesion_mask) : dict
        -_calculate_erythema_score(...) : float
        -_calculate_desquamation_score(...) : float
        -_calculate_induration_score(...) : float
        -_get_ai_baseline(pil_crop) : float
        -_finalize_scores(e, i, d) : dict
    }

    %% Database Module (Functional)
    class DatabaseModule {
        <<module>>
        +get_dynamodb_table()
        +save_analysis_to_db(analysis_result, user_id, image_filename)
        +save_questionnaire_to_db(assessment_data, assessment_id, user_id, timestamp)
        +get_user_assessments(user_id, limit)
        +get_assessment_by_timestamp(user_id, timestamp)
        -convert_to_decimal(obj)
    }

    %% API Routers
    class QuestionnaireRouter {
        <<module>>
        +generate_llm_assessment(data: QuestionnaireRequest) : dict
        -model: genai.GenerativeModel
    }

    class AnalyzeRouter {
        <<module>>
        +analyze_image_endpoint()
    }

    %% Data Schemas (Pydantic Models)
    class PsoriasisAnalysisResponse {
        +diagnosis: str
        +global_score: float
        +lesions_found: int
        +annotated_image_base64: str
        +details: List[LesionDetail]
        +db_metadata: DatabaseMetadata
    }

    class LesionDetail {
        +id: int
        +diagnosis: str
        +severity_score: float
        +area_pixels: int
        +erythema: int
        +induration: int
        +desquamation: int
    }

    class QuestionnaireRequest {
        +timestamp: str
        +screen1: Screen1
        +screen2: Screen2
        +screen3: Screen3
    }

    class QuestionnaireResponse {
        +assessment_id: str
        +timestamp: str
        +severity_assessment: str
        +psoriatic_arthritis_risk: str
        +nextSteps: List[str]
        +additionalNotes: str
        +treatment_urgency: str
        +recommended_followup_weeks: int
    }

    %% Relationships
    AnalyzeRouter ..> AIEngine : uses
    AnalyzeRouter ..> DatabaseModule : saves results
    AnalyzeRouter ..> PsoriasisAnalysisResponse : returns
    
    QuestionnaireRouter ..> QuestionnaireRequest : receives
    QuestionnaireRouter ..> QuestionnaireResponse : returns
    QuestionnaireRouter ..> DatabaseModule : saves results
    
    AIEngine ..> LesionDetail : creates
    PsoriasisAnalysisResponse *-- LesionDetail : contains
```

    ## Component Descriptions

    ### 1. AIEngine (Service Layer)
    The `AIEngine` is a singleton class responsible for the core computer vision and deep learning tasks.
    - **Responsibilities**:
        - Loading and managing AI models (YOLO "Sniper" and EfficientNet "Judge").
        - Preprocessing images (white balancing, cropping).
        - Calculating PASI metrics (Erythema, Induration, Desquamation).
        - Aggregating scores into a global severity diagnosis.

    ### 2. Database Module (Data Layer)
    A functional module handling all interactions with AWS DynamoDB.
    - **Responsibilities**:
        - Managing database connections.
        - Converting Python types to DynamoDB Decimal types.
        - Saving image analysis results.
        - Saving questionnaire assessments.
        - Retrieving user history.

    ### 3. API Routers (Controller Layer)
    - **AnalyzeRouter**: Handles image upload requests, invokes the `AIEngine`, and returns analysis results.
    - **QuestionnaireRouter**: Handles questionnaire submissions, invokes Google Gemini LLM for text analysis, and returns recommendations.

    ### 4. Schemas (Model Layer)
    Pydantic models that define the structure of API requests and responses, ensuring type safety and automatic validation.
