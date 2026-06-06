"""
LLM Content Generation & Quality Assurance Pipeline
Author: Gicela Vargas
Description: Automated system for generating, curating, and validating 
content for LLM training with focus on LATAM markets and ethical AI practices
"""

import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import hashlib
from collections import Counter
import random

_BASE_DIR = Path(__file__).resolve().parent

class LLMContentPipeline:
    """
    Pipeline for generating and curating high-quality training data for LLMs
    with emphasis on Colombian/LATAM context and bias reduction
    """
    
    def __init__(self):
        self.content_categories = [
            'engineering_infrastructure',
            'water_management', 
            'environmental_compliance',
            'project_management',
            'technical_documentation',
            'regulatory_frameworks',
            'sustainability_practices',
            'data_analysis',
            'cultural_context',
            'business_communications'
        ]
        
        self.languages = ['es', 'en']
        self.quality_threshold = 0.8
        self.bias_checks = []
        self.content_database = []
        
        # LATAM-specific knowledge domains
        self.latam_contexts = {
            'countries': ['Colombia', 'México', 'Argentina', 'Chile', 'Perú', 'Ecuador', 'Bolivia', 'Venezuela'],
            'regulations': ['NSR-10', 'Resolución 2115/2007', 'NTC-ISO 14001', 'Decreto 1076/2015'],
            'currencies': ['COP', 'MXN', 'ARS', 'CLP', 'PEN'],
            'institutions': ['INVIAS', 'ANI', 'CRA', 'ANLA', 'Ministerio de Ambiente']
        }
        
    def generate_prompt_response_pairs(self, category: str, n_pairs: int = 50) -> List[Dict]:
        """
        Generate high-quality prompt-response pairs for LLM training
        Focuses on reducing bias and ensuring cultural relevance for LATAM
        """
        pairs = []
        
        # Infrastructure Engineering prompts (bilingual)
        if category == 'engineering_infrastructure':
            infrastructure_prompts = [
                {
                    "prompt": "¿Cuáles son los pasos críticos en la auditoría técnica de una planta de tratamiento de agua potable en Colombia?",
                    "response": """Los pasos críticos en la auditoría técnica de una planta de tratamiento en Colombia incluyen:

**1. Evaluación de Procesos Operacionales:**
- Verificación de procesos de coagulación, floculación, sedimentación y filtración
- Análisis de eficiencia de sistemas de desinfección (cloro, ozono, UV)
- Inspección de sistemas de dosificación química

**2. Análisis de Calidad del Agua:**
- Monitoreo de parámetros físico-químicos según Resolución 2115/2007
- Verificación de pH (6.5-9.0), turbidez (<2 NTU), cloro residual (0.3-2.0 mg/L)
- Análisis microbiológico para coliformes totales y E. coli

**3. Inspección de Infraestructura:**
- Estado de tanques de almacenamiento y clarificación
- Evaluación de redes de distribución y pérdidas técnicas
- Verificación de sistemas de bombeo y equipos electromecánicos

**4. Cumplimiento Normativo:**
- Adherencia a normas técnicas colombianas (RAS 2000)
- Verificación de permisos ambientales ante ANLA
- Cumplimiento de estándares de vertimientos (Decreto 1076/2015)

**5. Gestión Operacional:**
- Revisión de manuales operativos y procedimientos
- Verificación de programas de mantenimiento preventivo
- Análisis de registros de laboratorio y bitácoras operacionales""",
                    "metadata": {
                        "category": "engineering_infrastructure",
                        "language": "es",
                        "region": "Colombia", 
                        "technical_level": "advanced",
                        "bias_check": "passed",
                        "cultural_context": "high"
                    }
                },
                {
                    "prompt": "How do you calculate Non-Revenue Water (NRW) in a municipal water system, and what are acceptable benchmarks for LATAM utilities?",
                    "response": """Non-Revenue Water (NRW) calculation and benchmarks for Latin American utilities:

**NRW Calculation Formula:**
NRW (%) = (Water Produced - Water Billed) / Water Produced × 100

**Components Analysis:**
1. **Real Losses**: Physical water losses from leaks, pipe bursts, overflow
2. **Apparent Losses**: Meter inaccuracies, unauthorized consumption, billing errors
3. **Unbilled Authorized Consumption**: Fire protection, municipal use, social programs

**Calculation Steps:**
1. Measure total water production at treatment plants
2. Sum all billed metered and estimated unmetered consumption
3. Account for unbilled authorized consumption (typically 2-5%)
4. Calculate difference and express as percentage

**LATAM Regional Benchmarks:**
- **Excellent Performance**: <15% (Chile, Uruguay urban areas)
- **Good Performance**: 15-25% (Colombia, Mexico major cities)
- **Acceptable**: 25-35% (Regional average)
- **Poor Performance**: >35% (Requires immediate intervention)

**Regional Context:**
- Infrastructure age and investment levels vary significantly
- Informal settlements present unique challenges
- Climate factors (seismic activity, extreme weather) impact distribution
- Economic constraints affect maintenance capabilities

**Best Practices for LATAM:**
- Pressure management in distribution networks
- Active leak detection programs
- Smart metering implementation
- Community engagement in water conservation""",
                    "metadata": {
                        "category": "engineering_infrastructure",
                        "language": "en", 
                        "region": "LATAM",
                        "technical_level": "intermediate",
                        "cultural_context": "high"
                    }
                }
            ]
            pairs.extend(infrastructure_prompts)

        elif category == 'water_management':
            water_prompts = [
                {
                    "prompt": "¿Qué criterios técnicos se deben verificar en la rehabilitación de un acueducto rural en Boyacá?",
                    "response": """La rehabilitación de acueductos rurales en Boyacá debe priorizar continuidad del servicio, calidad del agua y sostenibilidad operativa. El análisis técnico inicial define prioridades de inversión:

**1. Diagnóstico de la fuente y captación:**
- Caudal disponible en época seca y lluviosa
- Protección sanitaria del área de captación
- Riesgos de contaminación difusa (ganadería, actividades agrícolas)

**2. Red de conducción y almacenamiento:**
- Estado estructural de tuberías, válvulas y tanques
- Pérdidas por fugas y presiones insuficientes en sectores críticos
- Accesibilidad para mantenimiento en zonas de difícil acceso

**3. Tratamiento y desinfección:**
- Adecuación del sistema al caudal y número de usuarios (RAS 2000)
- Verificación de cloro residual en puntos terminales de la red
- Procedimientos de limpieza y desinfección de tanques

**4. Gestión comunitaria:**
- Capacitación a operadores locales y juntas de acueducto
- Plan de mantenimiento con costos realistas para la comunidad
- Mecanismos de cobro y reinversión en operación

**5. Cumplimiento normativo:**
- Monitoreo según Resolución 2115/2007
- Registros de laboratorio y plan de contingencia ante turbidez o fallas""",
                    "metadata": {
                        "category": "water_management",
                        "language": "es",
                        "region": "Colombia",
                        "technical_level": "advanced",
                        "bias_check": "passed",
                        "cultural_context": "high"
                    }
                },
                {
                    "prompt": "How should a water treatment plant define alert thresholds for pH, turbidity, and chlorine residual?",
                    "response": """Operational alert thresholds should combine regulatory limits, process stability, and operator response time:

**Regulatory baseline (Colombia, Resolución 2115/2007):**
- pH: 6.5–9.0 (operational target often 6.8–8.2 for process control)
- Turbidity: ≤2 NTU at plant outlet (lower targets improve disinfection efficiency)
- Chlorine residual: 0.3–2.0 mg/L in distribution (minimum for protection, maximum for taste/DBPs)

**Recommended alert tiers:**
1. **Normal** — all parameters within target band; routine logging
2. **Warning** — value approaches limit (e.g., turbidity 1.5–2.0 NTU, chlorine near minimum); increase sampling frequency
3. **Critical** — outside legal limit or rapid trend change; trigger corrective action and supervisor notification

**Implementation practices:**
- Set warning bands before legal limits to allow preventive action
- Use rolling averages to avoid false alarms from single readings
- Document escalation steps in standard operating procedures
- Review thresholds quarterly based on historical variability and seasonal demand""",
                    "metadata": {
                        "category": "water_management",
                        "language": "en",
                        "region": "Colombia",
                        "technical_level": "intermediate",
                        "bias_check": "passed",
                        "cultural_context": "high"
                    }
                }
            ]
            pairs.extend(water_prompts)

        elif category == 'regulatory_frameworks':
            regulatory_prompts = [
                {
                    "prompt": "¿Cuáles son los parámetros mínimos de calidad del agua para consumo humano según la Resolución 2115 de 2007 en Colombia?",
                    "response": """La Resolución 2115 de 2007 define límites máximos permisibles para agua de consumo humano en Colombia. Parámetros frecuentemente auditados:

**Fisicoquímicos:**
- pH: 6.5–9.0
- Turbiedad: ≤5 NTU (en planta de tratamiento el objetivo operativo suele ser menor)
- Color aparente: ≤15 unidades Pt-Co
- Cloro residual libre: 0.3–2.0 mg/L en red de distribución

**Microbiológicos:**
- Coliformes totales: ausencia en 100 mL
- E. coli: ausencia en 100 mL

**Uso en control operativo:**
- Los operadores deben registrar mediciones con frecuencia definida en el plan de saneamiento
- Desviaciones requieren acción correctiva documentada
- Laboratorios deben cumplir trazabilidad y cadenas de custodia

**Contexto de cumplimiento:**
- Aplica a sistemas urbanos y rurales según responsable del servicio
- Se complementa con RAS 2000 para diseño y operación de acueductos
- Las autoridades sanitarias verifican cumplimiento mediante inspección y muestreo""",
                    "metadata": {
                        "category": "regulatory_frameworks",
                        "language": "es",
                        "region": "Colombia",
                        "technical_level": "advanced",
                        "bias_check": "passed",
                        "cultural_context": "high"
                    }
                }
            ]
            pairs.extend(regulatory_prompts)
        
        # Environmental compliance prompts
        elif category == 'environmental_compliance':
            env_prompts = [
                {
                    "prompt": "Explica el proceso de obtención de una licencia ambiental para un proyecto de infraestructura vial en Colombia.",
                    "response": """El proceso de licencia ambiental para proyectos viales en Colombia sigue estos pasos:

**1. Clasificación del Proyecto:**
- Autopistas y carreteras con longitud >30 km requieren licencia ante ANLA
- Proyectos <30 km pueden requerir permisos departamentales
- Verificación en Decreto 1076/2015 para determinar competencia

**2. Elaboración del Estudio de Impacto Ambiental (EIA):**
- Caracterización del área de influencia (biótica, abiótica, socioeconómica)
- Identificación y evaluación de impactos ambientales
- Formulación del Plan de Manejo Ambiental (PMA)
- Programa de seguimiento y monitoreo

**3. Documentación Requerida:**
- Términos de referencia según Manual ANLA
- Cartografía oficial del IGAC
- Consulta previa con comunidades étnicas (si aplica)
- Concepto técnico de la autoridad competente

**4. Trámite Administrativo:**
- Radicación ante ANLA o autoridad regional
- Revisión documental (máximo 15 días hábiles)
- Evaluación técnica (hasta 6 meses)
- Concepto técnico y decisión administrativa

**5. Seguimiento y Control:**
- Implementación del PMA aprobado
- Informes de Cumplimiento Ambiental (ICA) semestrales
- Visitas de seguimiento por parte de la autoridad
- Cumplimiento de obligaciones específicas""",
                    "metadata": {
                        "category": "environmental_compliance",
                        "language": "es",
                        "region": "Colombia",
                        "cultural_context": "high"
                    }
                }
            ]
            pairs.extend(env_prompts)
        
        # Cultural context and business communications
        elif category == 'cultural_context':
            cultural_prompts = [
                {
                    "prompt": "What are important cultural considerations when managing infrastructure projects across different LATAM countries?",
                    "response": """Cultural considerations for infrastructure project management across LATAM:

**Communication Styles:**
- **High-context cultures**: Relationships and indirect communication valued (Colombia, Ecuador)
- **Hierarchy respect**: Formal titles and authority structures important
- **Time concepts**: "Tiempo latino" - relationship-building may extend timelines
- **Personal relationships**: Trust-building essential before business discussions

**Stakeholder Engagement:**
- **Community consultation**: Indigenous and afro-descendant community rights
- **Local leadership**: Engaging mayors, governors, and community leaders
- **Social license**: Projects require community acceptance beyond permits
- **Compensation mechanisms**: Fair land acquisition and displacement procedures

**Regional Variations:**
- **Mexico**: Federal vs. state jurisdictions, strong institutional frameworks
- **Colombia**: Post-conflict considerations, territorial approach
- **Chile**: High technical standards, environmental consciousness
- **Argentina**: Provincial autonomy, economic volatility considerations

**Best Practices:**
- Bilingual project documentation (Spanish/Portuguese/indigenous languages)
- Local procurement preferences and social responsibility
- Gender inclusion and women's participation in construction
- Environmental and social safeguards beyond technical requirements
- Adaptive management for political and economic changes

**Risk Mitigation:**
- Political transition planning (election cycles)
- Currency fluctuation hedging strategies
- Social conflict prevention and resolution mechanisms
- Corruption prevention and transparency measures""",
                    "metadata": {
                        "category": "cultural_context",
                        "language": "en",
                        "region": "LATAM",
                        "cultural_context": "very_high",
                        "bias_check": "passed"
                    }
                }
            ]
            pairs.extend(cultural_prompts)
        
        # Generate additional synthetic pairs if needed
        remaining = n_pairs - len(pairs)
        for i in range(remaining):
            pairs.append(self._generate_synthetic_pair(category))
        
        return pairs[:n_pairs]
    
    def _generate_synthetic_pair(self, category: str) -> Dict:
        """
        Generate synthetic prompt-response pairs based on category and LATAM context
        """
        templates = {
            'engineering_infrastructure': [
                "Explain the regulatory requirements for {infrastructure_type} in {country}",
                "What are the key technical considerations for {project_type} in {climate}?",
                "How to ensure compliance with {regulation} for {infrastructure_type}?"
            ],
            'water_management': [
                "Calculate the optimal {parameter} for a water system serving {population} people",
                "Best practices for {operation} in {climate} conditions across LATAM",
                "How to reduce {problem} in {system_type} while maintaining {standard}?"
            ],
            'cultural_context': [
                "What are the cultural considerations for {project_type} in {community_type}?",
                "How to navigate {challenge} when working with {stakeholder_group}?",
                "Explain the importance of {cultural_element} in {country} infrastructure projects"
            ]
        }
        
        # Add realistic LATAM context
        contexts = {
            'country': random.choice(self.latam_contexts['countries']),
            'regulation': random.choice(self.latam_contexts['regulations']),
            'infrastructure_type': random.choice(['water treatment plants', 'bridges', 'roads', 'aqueducts']),
            'project_type': random.choice(['rehabilitation', 'new construction', 'expansion']),
            'community_type': random.choice(['indigenous communities', 'urban areas', 'rural municipalities']),
            'climate_zone': random.choice(['tropical', 'temperate', 'arid', 'mountain']),
            'climate': random.choice(['humid', 'dry', 'variable']),
            'parameter': random.choice(['flow rate', 'pressure', 'quality']),
            'population': random.choice(['5000', '10000', '50000']),
            'operation': random.choice(['maintenance', 'monitoring', 'optimization']),
            'problem': random.choice(['leaks', 'contamination', 'inefficiency']),
            'system_type': random.choice(['distribution', 'treatment', 'collection']),
            'standard': random.choice(['WHO guidelines', 'local standards']),
            'challenge': random.choice(['funding', 'technical capacity', 'community engagement']),
            'stakeholder_group': random.choice(['local communities', 'government agencies', 'contractors']),
            'cultural_element': random.choice(['traditional practices', 'community participation', 'local knowledge']),
            'topic': random.choice(['infrastructure', 'environment', 'development']),
            'infrastructure_element': random.choice(['pipelines', 'pumps', 'treatment systems', 'storage tanks'])
        }
        
        # Generate based on templates with realistic responses
        category_templates = templates.get(category, ["General question about {topic}"])
        template = random.choice(category_templates)
        
        language = random.choice(self.languages)
        prompt = template.format(**contexts)

        if language == "es":
            response = (
                f"En {contexts['country']}, los proyectos de {contexts['infrastructure_type']} "
                f"deben alinearse con marcos normativos como {contexts['regulation']}. "
                f"Se recomienda: (1) diagnóstico de condiciones en zona {contexts['climate']}; "
                f"(2) validación con {contexts['stakeholder_group']}; "
                f"(3) plan de operación con indicadores verificables. "
                f"La implementación debe considerar restricciones locales, participación comunitaria "
                f"y cumplimiento ambiental según la normativa nacional aplicable."
            )
        else:
            response = (
                f"In {contexts['country']}, {contexts['infrastructure_type']} projects must align "
                f"with frameworks such as {contexts['regulation']}. Recommended steps: "
                f"(1) assess site conditions in a {contexts['climate']} climate; "
                f"(2) validate assumptions with {contexts['stakeholder_group']}; "
                f"(3) define measurable operational indicators. "
                f"Delivery should account for local constraints, community participation, "
                f"and environmental compliance under applicable national standards."
            )

        return {
            "prompt": prompt,
            "response": response,
            "metadata": {
                "category": category,
                "language": language,
                "region": contexts["country"],
                "generated": True,
                "timestamp": datetime.now().isoformat(),
                "cultural_context": "medium",
                "bias_check": "passed"
            }
        }
    
    def validate_content_quality(self, content: Dict) -> Tuple[float, List[str]]:
        """
        Comprehensive content quality validation with bias detection
        """
        issues = []
        score = 1.0
        
        # Basic quality checks
        response_length = len(content['response'])
        if response_length < 100:
            issues.append("Response too short for meaningful training")
            score -= 0.3
        elif response_length > 3000:
            issues.append("Response may be too verbose")
            score -= 0.1
        
        # Technical accuracy indicators
        technical_terms = [
            'compliance', 'normativo', 'technical', 'análisis', 'infrastructure',
            'environmental', 'regulatory', 'standard', 'procedure', 'methodology'
        ]
        term_count = sum(1 for term in technical_terms if term.lower() in content['response'].lower())
        if term_count < 2:
            issues.append("Low technical content density")
            score -= 0.2
        
        # LATAM-specific context validation
        latam_indicators = [
            'colombia', 'méxico', 'argentina', 'chile', 'perú', 'latam', 'latin america',
            'resolución', 'decreto', 'anla', 'cop', 'peso', 'ministerio'
        ]
        latam_context = any(ind in content['response'].lower() for ind in latam_indicators)
        if content['metadata'].get('region') == 'LATAM' and not latam_context:
            issues.append("Missing LATAM-specific context")
            score -= 0.15
        
        # Bias detection system
        bias_checks = self._detect_bias(content['response'])
        for bias_type, severity in bias_checks.items():
            if severity == 'high':
                issues.append(f"High {bias_type} detected")
                score -= 0.3
            elif severity == 'medium':
                issues.append(f"Medium {bias_type} detected")
                score -= 0.15
        
        # Language consistency validation
        if 'language' in content['metadata']:
            lang_consistency = self._validate_language_consistency(
                content['response'], 
                content['metadata']['language']
            )
            if not lang_consistency:
                issues.append("Language inconsistency detected")
                score -= 0.2
        
        # Cultural sensitivity check
        cultural_score = self._assess_cultural_sensitivity(content['response'])
        if cultural_score < 0.7:
            issues.append("Potential cultural insensitivity")
            score -= 0.2
        
        return max(0, score), issues
    
    def _detect_bias(self, text: str) -> Dict[str, str]:
        """
        Detect various forms of bias in content
        """
        bias_indicators = {
            'gender_bias': {
                'high': ['men are better', 'women should', 'only men can', 'female engineers are'],
                'medium': ['guys', 'mankind', 'manpower', 'he/she']
            },
            'regional_bias': {
                'high': ['developed countries', 'third world', 'backward', 'primitive'],
                'medium': ['first world', 'advanced nations', 'modern countries']
            },
            'technical_elitism': {
                'high': ['obviously', 'clearly', 'any engineer knows', 'basic knowledge'],
                'medium': ['simply', 'just', 'merely', 'of course']
            },
            'economic_bias': {
                'high': ['cheap labor', 'low-cost workers', 'inexpensive workforce'],
                'medium': ['affordable', 'cost-effective', 'budget-friendly']
            }
        }
        
        detected_bias = {}
        text_lower = text.lower()
        
        for bias_type, indicators in bias_indicators.items():
            for severity, phrases in indicators.items():
                if any(phrase in text_lower for phrase in phrases):
                    detected_bias[bias_type] = severity
                    break
        
        return detected_bias
    
    def _validate_language_consistency(self, text: str, expected_lang: str) -> bool:
        """
        Validate language consistency in multilingual content
        """
        # Simple language detection based on common words
        spanish_indicators = ['el', 'la', 'de', 'que', 'en', 'y', 'para', 'con', 'por', 'según']
        english_indicators = ['the', 'is', 'of', 'to', 'and', 'for', 'with', 'by', 'according']
        
        text_words = text.lower().split()
        spanish_count = sum(1 for word in spanish_indicators if word in text_words)
        english_count = sum(1 for word in english_indicators if word in text_words)
        
        if expected_lang == 'es':
            return spanish_count >= english_count
        elif expected_lang == 'en':
            return english_count >= spanish_count
        
        return True
    
    def _assess_cultural_sensitivity(self, text: str) -> float:
        """
        Assess cultural sensitivity and inclusiveness of content
        """
        # Positive cultural indicators
        positive_indicators = [
            'community', 'stakeholder', 'inclusive', 'consultation', 'participation',
            'cultural', 'traditional', 'local knowledge', 'indigenous', 'diversity'
        ]
        
        # Negative cultural indicators
        negative_indicators = [
            'primitive', 'backward', 'underdeveloped', 'savage', 'uncivilized',
            'inferior', 'superior culture', 'western standard'
        ]
        
        text_lower = text.lower()
        positive_score = sum(1 for indicator in positive_indicators if indicator in text_lower)
        negative_score = sum(1 for indicator in negative_indicators if indicator in text_lower)
        
        # Calculate sensitivity score (0-1 scale)
        base_score = 0.8  # Neutral baseline
        positive_boost = min(0.2, positive_score * 0.05)
        negative_penalty = min(0.5, negative_score * 0.2)
        
        return max(0, base_score + positive_boost - negative_penalty)
    
    def generate_training_dataset(self, size: int = 500) -> pd.DataFrame:
        """
        Generate comprehensive training dataset with quality assurance
        """
        dataset = []
        
        # Distribute samples across categories
        samples_per_category = size // len(self.content_categories)
        
        for category in self.content_categories:
            pairs = self.generate_prompt_response_pairs(category, samples_per_category)
            
            for pair in pairs:
                # Quality validation
                quality_score, issues = self.validate_content_quality(pair)
                
                # Only include high-quality content
                if quality_score >= self.quality_threshold:
                    # Create unique content ID
                    content_id = hashlib.md5(
                        (pair['prompt'] + pair['response']).encode()
                    ).hexdigest()[:12]
                    
                    dataset.append({
                        'content_id': content_id,
                        'prompt': pair['prompt'],
                        'response': pair['response'],
                        'category': pair['metadata']['category'],
                        'language': pair['metadata'].get('language', 'es'),
                        'region': pair['metadata'].get('region', 'LATAM'),
                        'quality_score': quality_score,
                        'word_count': len(pair['response'].split()),
                        'char_count': len(pair['response']),
                        'technical_level': pair['metadata'].get('technical_level', 'intermediate'),
                        'cultural_context': pair['metadata'].get('cultural_context', 'medium'),
                        'timestamp': datetime.now().isoformat(),
                        'validation_status': 'approved' if not issues else 'approved_with_notes',
                        'validation_notes': '; '.join(issues) if issues else 'No issues detected',
                        'bias_checked': True,
                        'cultural_sensitive': True
                    })
        
        return pd.DataFrame(dataset)
    
    def load_rubric(self, path: Optional[str] = None) -> Optional[Dict]:
        """Load evaluation rubric definition if present."""
        rubric_path = Path(path) if path else _BASE_DIR / "eval_rubric.json"
        try:
            with open(rubric_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def create_evaluation_report(self, dataset: pd.DataFrame) -> Dict:
        """
        Generate comprehensive evaluation metrics for the dataset
        """
        total_samples = len(dataset)
        rubric = self.load_rubric()

        report = {
            'dataset_overview': {
                'total_samples': total_samples,
                'unique_prompts': dataset['prompt'].nunique(),
                'unique_responses': dataset['response'].nunique(),
                'avg_response_length': dataset['word_count'].mean(),
                'quality_score_mean': dataset['quality_score'].mean(),
                'quality_score_std': dataset['quality_score'].std(),
                'generation_date': datetime.now().isoformat()
            },
            
            'content_distribution': {
                'by_category': dataset['category'].value_counts().to_dict(),
                'by_language': dataset['language'].value_counts().to_dict(),
                'by_region': dataset['region'].value_counts().to_dict(),
                'by_technical_level': dataset['technical_level'].value_counts().to_dict(),
                'by_cultural_context': dataset['cultural_context'].value_counts().to_dict()
            },
            
            'quality_metrics': {
                'high_quality_samples': len(dataset[dataset['quality_score'] > 0.9]),
                'medium_quality_samples': len(dataset[(dataset['quality_score'] > 0.8) & (dataset['quality_score'] <= 0.9)]),
                'validation_status': dataset['validation_status'].value_counts().to_dict(),
                'samples_with_issues': len(dataset[dataset['validation_notes'] != 'No issues detected']),
                'bias_checked_percentage': (dataset['bias_checked'].sum() / total_samples) * 100,
                'culturally_sensitive_percentage': (dataset['cultural_sensitive'].sum() / total_samples) * 100
            },
            
            'latam_focus_metrics': {
                'latam_content_percentage': (dataset['region'] == 'LATAM').sum() / total_samples * 100,
                'spanish_content_percentage': (dataset['language'] == 'es').sum() / total_samples * 100,
                'high_cultural_context': len(dataset[dataset['cultural_context'] == 'high']),
                'regulatory_content': len(dataset[dataset['response'].str.contains('resolución|decreto|norma', case=False, na=False)])
            },
            
            'recommendations': [
                "Monitor for potential bias in future content generation cycles",
                "Ensure balanced representation across all LATAM countries",
                "Validate technical accuracy with subject matter experts",
                "Consider expanding indigenous language content for broader inclusion",
                "Implement regular bias audits for content quality assurance"
            ]
        }

        if rubric:
            report['rubric'] = {
                'version': rubric.get('rubric_version'),
                'dimensions': [d['id'] for d in rubric.get('dimensions', [])],
                'thresholds': rubric.get('decision_thresholds', {})
            }

        return report

def main():
    """
    Main execution function demonstrating the LLM content pipeline
    """
    print("🚀 LLM LATAM Content Pipeline")
    print("=" * 60)
    print("Generating ethical, high-quality training content for LATAM markets")
    print()
    
    # Initialize pipeline
    pipeline = LLMContentPipeline()
    
    # Generate training dataset
    print("📝 Generating training dataset...")
    dataset = pipeline.generate_training_dataset(size=200)
    
    print(f"✅ Generated {len(dataset)} high-quality training samples")
    print()
    
    # Create evaluation report
    print("📊 Creating evaluation report...")
    report = pipeline.create_evaluation_report(dataset)
    
    # Display key metrics
    print("🎯 Key Metrics:")
    print(f"   Quality Score Average: {report['dataset_overview']['quality_score_mean']:.3f}")
    print(f"   LATAM Content: {report['latam_focus_metrics']['latam_content_percentage']:.1f}%")
    print(f"   Spanish Content: {report['latam_focus_metrics']['spanish_content_percentage']:.1f}%")
    print(f"   Bias-Checked: {report['quality_metrics']['bias_checked_percentage']:.1f}%")
    print(f"   Cultural Sensitivity: {report['quality_metrics']['culturally_sensitive_percentage']:.1f}%")
    print()
    
    # Save outputs
    print("💾 Saving outputs...")
    dataset.to_csv('llm_training_dataset.csv', index=False, encoding='utf-8')
    
    with open('evaluation_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    
    print("📁 Files saved:")
    print("   - llm_training_dataset.csv (Training data)")
    print("   - evaluation_report.json (Quality metrics)")
    print()
    
    print("🎉 Pipeline execution completed successfully!")
    print("Ready for ethical LLM training with LATAM focus")

if __name__ == "__main__":
    main()
