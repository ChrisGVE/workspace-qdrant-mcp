# Document Format Extraction Benchmark Report

**Generated:** 2025-08-30 19:58:33

## Executive Summary

- **Total formats tested:** 7
- **Successful extractions:** 6
- **Failed extractions:** 1

## Library Availability

- **pypdf:** ✅ Available
- **pdfplumber:** ✅ Available
- **pymupdf:** ❌ Not Available
- **ebooklib:** ✅ Available
- **beautifulsoup4:** ✅ Available
- **textract:** ❌ Not Available
- **markdown:** ✅ Available
- **python-docx:** ✅ Available

## Format Quality Rankings

| Rank | Format | Quality Score | Extraction Time | Completeness | Accuracy | Similarity |
|------|--------|---------------|-----------------|--------------|-----------|------------|
| 1 | txt | 1.000 | 0.000s | 1.000 | 1.000 | 1.000 |
| 2 | md | 1.000 | 0.000s | 1.000 | 1.000 | 1.000 |
| 3 | html | 0.906 | 0.003s | 0.863 | 0.932 | 0.938 |
| 4 | epub | 0.906 | 0.004s | 0.863 | 0.932 | 0.938 |
| 5 | pdf_pypdf | 0.811 | 0.010s | 0.821 | 0.897 | 0.712 |
| 6 | pdf_pdfplumber | 0.744 | 0.038s | 0.768 | 0.839 | 0.618 |

## Format Precedence Rules

Based on the benchmark results, the following precedence rules are recommended:

1. **txt** - ✅ Recommended
   - Quality Score: 1.000
   - Extraction Time: 0.000s

2. **md** - ✅ Recommended
   - Quality Score: 1.000
   - Extraction Time: 0.000s

3. **html** - ✅ Recommended
   - Quality Score: 0.906
   - Extraction Time: 0.003s

4. **epub** - ✅ Recommended
   - Quality Score: 0.906
   - Extraction Time: 0.004s

5. **pdf_pypdf** - ✅ Recommended
   - Quality Score: 0.811
   - Extraction Time: 0.010s

6. **pdf_pdfplumber** - ✅ Recommended
   - Quality Score: 0.744
   - Extraction Time: 0.038s

## Implementation Recommendations

- Primary recommendation: txt (quality: 1.000)
- Fastest extraction: txt (0.000s)

## Detailed Results

### txt

- **Status:** ✅ Success
- **Content Length:** 970 characters
- **Word Count:** 145 words
- **Extraction Time:** 0.000 seconds
- **Completeness:** 1.000
- **Accuracy:** 1.000
- **Similarity:** 1.000

### md

- **Status:** ✅ Success
- **Content Length:** 970 characters
- **Word Count:** 145 words
- **Extraction Time:** 0.000 seconds
- **Completeness:** 1.000
- **Accuracy:** 1.000
- **Similarity:** 1.000

### html

- **Status:** ✅ Success
- **Content Length:** 784 characters
- **Word Count:** 123 words
- **Extraction Time:** 0.003 seconds
- **Completeness:** 0.863
- **Accuracy:** 0.932
- **Similarity:** 0.938

### pdf_pypdf

- **Status:** ✅ Success
- **Content Length:** 764 characters
- **Word Count:** 118 words
- **Extraction Time:** 0.010 seconds
- **Completeness:** 0.821
- **Accuracy:** 0.897
- **Similarity:** 0.712

### pdf_pdfplumber

- **Status:** ✅ Success
- **Content Length:** 810 characters
- **Word Count:** 118 words
- **Extraction Time:** 0.038 seconds
- **Completeness:** 0.768
- **Accuracy:** 0.839
- **Similarity:** 0.618

### pdf_pymupdf

- **Status:** ❌ Failed
- **Error:** pymupdf not available

### epub

- **Status:** ✅ Success
- **Content Length:** 826 characters
- **Word Count:** 125 words
- **Extraction Time:** 0.004 seconds
- **Completeness:** 0.863
- **Accuracy:** 0.932
- **Similarity:** 0.938

