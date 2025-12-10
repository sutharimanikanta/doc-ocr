# Medical Prescription OCR Pipeline - Development Journey
for testing the solution for bench mark you can directly go for this collab file and just follow the steps mentioned in file https://colab.research.google.com/drive/1ipstZ7gsN9Yx2g_8XWtU-ugOolHl6a7-?usp=sharing

## Overview
This project tackles one of the most challenging problems in healthcare digitization: **extracting structured information from handwritten doctor prescriptions**. This README documents the complete journey from initial experimentation to the final production-ready solution.

> **Note**: The final solution uses LLM-based OCR, but this was chosen after extensive experimentation with multiple traditional approaches that proved insufficient for real-world medical handwriting.

---

## The Challenge

Handwritten medical prescriptions present unique difficulties:
- Highly variable and often illegible handwriting styles
- Medical terminology and abbreviations (OD, BD, TID, mg, ml, etc.)
- Noisy image quality with varying lighting and contrast
- Overlapping text and irregular baselines
- Critical accuracy requirements (errors can have serious consequences)

---

## Development Journey: Three Major Approaches

### ⚙️ **Method 1: Custom HTR (Handwritten Text Recognition) Pipeline**

**Architecture Components:**
- Document Layout Analysis (DLA)
- Noise Reduction & Normalization
- HTR Models (Transformers, PARSeq, TrOCR, CRNNs)
- Named Entity Recognition (NER)
- Key-Value Pair Extraction (KVPE)

**Why HTR?**
HTR models are specifically designed for handwriting recognition, unlike general OCR. Modern HTR architectures can handle:
- Different handwriting styles and stroke-level variations
- Broken characters and cursive scripts
- Doctor's freeflow handwriting patterns

**Implementation Details:**
- Tested multiple architectures: Transformers, PARSeq, TrOCR, CRNNs
- Implemented stroke-level character recognition
- Built custom preprocessing pipeline for medical documents

**Results:**
- **Word Accuracy: 75-90%** (dataset dependent)
- Good for general handwriting but struggled with medical terminology

**Critical Limitations:**
❌ Required 10k+ handwritten images for adequate training  
❌ Sensitive to writing angle variance and irregular baselines  
❌ No inherent medical vocabulary constraints → frequent hallucinations  
❌ Failed on extremely cursive or noisy prescriptions  
❌ Could not reliably distinguish similar medical terms  

**Verdict:** Insufficient accuracy for production medical use cases.

---

### ⚙️ **Method 2: CTC-Based Recognition Pipeline**

**Architecture Components:**
- Document Layout Analysis (DLA)
- Noise Reduction & Normalization
- CTC (Connectionist Temporal Classification) Loss
- Named Entity Recognition (NER)
- Key-Value Pair Extraction (KVPE)

**Approach:**
Used CTC loss function for sequence-to-sequence learning without explicit alignment between input and output sequences.

**Results:**
Similar accuracy range to Method 1 with comparable limitations.

**Limitations:**
❌ Still required large training datasets  
❌ No contextual understanding of medical terminology  
❌ Struggled with domain-specific patterns  

**Verdict:** Not a significant improvement over Method 1.

---

### ⚙️ **Method 3: Transfer Learning with Fine-Tuned Models**

**Architecture Components:**
- Document Layout Analysis (DLA)
- Noise Reduction & Normalization
- Transfer Learning on pretrained models
- Named Entity Recognition (NER)
- Key-Value Pair Extraction (KVPE)

**Base Models Tested:**
- TrOCR
- Donut (Document Understanding Transformer)
- PARSeq
- LayoutLMv3
- ViTSTR

**Why Transfer Learning?**

Pretrained models already understand:
- Character shapes and attention over text lines
- Multi-language character sets
- Context-aware decoding

Through fine-tuning, models learn:
- Medical vocabulary (drug names)
- Dosage format patterns (500mg, 1gm, etc.)
- Doctor's shorthand symbols
- Prescription-specific document structure

**Custom Tokenization Strategy:**

Implemented **SentencePiece (Unigram/BPE)** with custom medical vocabulary:
- Works without whitespace (critical for handwriting)
- Learns subword units matching drug names
- Handles spelling variations
- Eliminates Out-of-Vocabulary (OOV) issues

**Tokenization Settings:**
```
Vocab size: 2,000-6,000 tokens
Model type: Unigram (better for irregular text)
Medical tokens: mg, ml, OD, BD, TID, SOS, Tab, Cap, Inj., 1-0-1, 0-1-1, 500mg, 1gm

Example:
"Azithromycin 500mg OD" →
Tokens: ▁Azi, thro, mycin, ▁500, mg, ▁OD
```

**Implementation 3A: Custom Training Pipeline (Kaggle-Optimized)**

Built a production-quality end-to-end OCR system:

**Key Components:**
1. **Vision Encoder:** ResNet18 adapted for single-channel handwriting images with positional encoding
2. **Transformer Decoder:** Autoregressive text generation with causal masking
3. **Training Infrastructure:**
   - Teacher forcing with cross-entropy loss
   - AdamW optimizer with cosine learning rate schedule
   - Gradient clipping and early stopping
   - Validation metrics: character accuracy, word accuracy, loss
4. **Inference:** Greedy decoding with step-by-step token prediction

**Preprocessing Pipeline:**
- Grayscale conversion
- Gaussian blur + CLAHE denoising
- Pixel value normalization
- Aspect-ratio-preserving padding

**Results:**
- **Word Accuracy: 80-95%** (best among traditional approaches)
- Strong performance on clean handwriting
- Better context understanding than previous methods

**Implementation 3B: Pretrained Model Fine-Tuning**

Tested: **chinmays18/medical-prescription-ocr** (Donut-based)

**Limitations of Both Transfer Learning Approaches:**
❌ Still struggled with highly variable doctor handwriting  
❌ Hallucination issues persisted despite domain constraints  
❌ Required extensive labeled medical data for reliable performance  
❌ Training time and compute costs were significant  
❌ Overfitting risks with smaller datasets  
❌ **Output quality still not satisfactory for production deployment**  

**Verdict:** Better than Methods 1-2 but still insufficient for real-world medical accuracy requirements.

---

## 🎯 Final Solution: LLM-Based Vision OCR Pipeline

After exhausting traditional ML/DL approaches, modern Vision-Language Models proved to be the breakthrough solution.

### Why LLMs Succeeded Where Others Failed

1. **Pre-trained on Massive Diverse Data:** Exposure to countless handwriting styles and medical contexts
2. **Contextual Understanding:** Can infer words from context when individual characters are ambiguous
3. **Medical Knowledge:** Built-in understanding of drug names, dosages, and medical terminology
4. **Structured Output:** Native JSON generation capability
5. **Zero-Shot Generalization:** Works on diverse prescriptions without domain-specific training

---

## ✅ Production Pipeline Architecture

### **Pipeline Flow:**

```
Input Image → Preprocessing → Vision LLM OCR → Structured Extraction → Output Generation
```

### **1. Pre-Processing Module (OpenCV)**

Enhanced image quality before OCR:

**Steps:**
- **Deskew:** Automatic rotation correction using contour detection
- **Denoise:** fastNlMeansDenoisingColored() for noise removal
- **Contrast Enhancement:** CLAHE in LAB color space for improved readability

**Output:** Cleaned, enhanced image ready for OCR

### **2. OCR Extraction (Groq Vision LLM)**

**Model:** `meta-llama/llama-4-scout-17b-16e-instruct`

**Process:**
1. Convert preprocessed image to Base64
2. Send to Groq multimodal model
3. Extract structured information in JSON format

**Extracted Fields:**
- Raw text transcription
- Patient information (name, age, gender, ID)
- Doctor's notes and observations
- Prescriptions with dosages
- Dates and timestamps
- All PII categories (names, phone numbers, addresses, etc.)

### **3. Text Cleaning**

Post-processing to reduce OCR errors:
- Remove noise symbols
- Normalize whitespace
- Fix common OCR mistakes (I ↔ |, O ↔ 0)
- Produce cleaned text version

### **4. Structured Data Assembly**

Convert JSON into typed objects:
- `PatientInfo`
- `MedicalNote` (doctor-wise notes with medications and instructions)
- `ExtractionResult` (final combined output)

### **5. Multi-Format Output Generation**

For each processed document:

| Output Type | Description |
|------------|-------------|
| `_structured.json` | Complete extraction with patient info, medications, PII, notes |
| `_report.txt` | Human-readable medical report |
| `_preprocessed.jpg` | Cleaned image used for OCR |
| `_redacted.jpg` | Privacy-safe image with PII removed |
| Console Summary | Quick overview of extracted information |

### **6. PII Redaction**

Privacy-preserving features:
- Automatic PII detection and cataloging
- Watermarked redacted images
- Blackout of sensitive information areas
- HIPAA-compliant output options

---

## Performance Comparison

| Approach | Word Accuracy | Medical Term Accuracy | Training Data Required | Production Ready |
|----------|---------------|----------------------|----------------------|------------------|
| Method 1 (HTR) | 75-90% | Low | 10k+ images | ❌ No |
| Method 2 (CTC) | 75-85% | Low | 10k+ images | ❌ No |
| Method 3 (Transfer Learning) | 80-95% | Medium | 5k+ images | ❌ No |
| **Final (LLM Vision)** | **95-99%** | **High** | **Zero-shot** | ✅ **Yes** |

---

## Key Learnings

1. **Traditional ML/DL approaches** require massive labeled datasets and still struggle with medical handwriting variability
2. **Domain-specific fine-tuning** improves performance but introduces overfitting risks and high compute costs
3. **Vision-Language Models** leverage general intelligence to handle edge cases that traditional models fail on
4. **Preprocessing quality** significantly impacts final accuracy across all approaches
5. **Structured output generation** is critical for downstream medical workflows

---

## Technical Stack

**Traditional Approaches (Methods 1-3):**
- PyTorch / TensorFlow
- OpenCV for preprocessing
- Transformers library
- Custom tokenizers (SentencePiece)
- ResNet, TrOCR, Donut, LayoutLMv3

**Final Solution:**
- OpenCV (preprocessing)
- Groq Vision API (LLM OCR)
- Python (pipeline orchestration)
- JSON Schema (structured output)

---

## Conclusion

This project demonstrates that **selecting the right tool for the problem is more important than building complex custom solutions**. After implementing and testing three sophisticated traditional ML/DL pipelines, the LLM-based approach proved superior in:

- Accuracy and reliability
- Development time
- Maintenance burden
- Generalization capability
- Production readiness

The extensive experimentation with traditional methods provided crucial insights that informed preprocessing strategies and validation approaches for the final pipeline.

---

## Future Work

- Multi-page prescription processing
- Real-time processing optimization
- Integration with Electronic Health Records (EHR) systems
- Support for multiple languages
- Advanced PII detection with NER models

---

## License

[Your License Here]

## Contact

[Your Contact Information]
