#!/usr/bin/env python3
"""
Document Validation Script for COMP4702 Assignment

Validates the DOCUMENT.md file for completeness, structure, and academic format.
Checks word count, section presence, figure references, and content quality.
"""

import re
import os
import sys
from pathlib import Path


def validate_abstract(content):
    """Validate abstract word count and presence"""
    print("\n" + "="*50)
    print("ABSTRACT VALIDATION")
    print("="*50)
    
    abstract_match = re.search(r'## Abstract\s+([\s\S]+?)\s*## ', content)
    if abstract_match:
        abstract = abstract_match.group(1).strip()
        # Remove markdown formatting for accurate word count
        clean_abstract = re.sub(r'\*\*.*?\*\*', '', abstract)  # Remove bold
        clean_abstract = re.sub(r'`.*?`', '', clean_abstract)   # Remove code
        clean_abstract = re.sub(r'\[.*?\]\(.*?\)', '', clean_abstract)  # Remove links
        
        word_count = len(clean_abstract.split())
        print(f"Abstract word count: {word_count}")
        
        if word_count == 150:
            print("✓ Abstract is exactly 150 words")
            return True
        elif 145 <= word_count <= 155:
            print(f"⚠️  Abstract is close to 150 words (within 5 words)")
            return True
        else:
            print(f"❌ Abstract should be exactly 150 words (current: {word_count})")
            return False
    else:
        print("❌ Abstract section not found!")
        return False


def validate_sections(content):
    """Validate presence of required sections"""
    print("\n" + "="*50)
    print("SECTION STRUCTURE VALIDATION")
    print("="*50)
    
    required_sections = [
        "Abstract",
        "Introduction", 
        "Dataset & Pre-processing",
        "Modeling Methodology",
        "Evaluation & Results", 
        "Discussion",
        "Conclusion & Future Work",
        "References",
        "Appendix"
    ]
    
    missing_sections = []
    found_sections = []
    
    for section in required_sections:
        # Look for both ## Section and # Section formats
        pattern = rf'#{1,2}\s+{re.escape(section)}'
        if re.search(pattern, content, re.IGNORECASE):
            print(f"✓ Section '{section}' found")
            found_sections.append(section)
        else:
            print(f"❌ Section '{section}' NOT found!")
            missing_sections.append(section)
    
    print(f"\nSection summary: {len(found_sections)}/{len(required_sections)} sections found")
    
    if missing_sections:
        print("\nMissing sections:")
        for section in missing_sections:
            print(f"  - {section}")
        return False
    else:
        print("✅ All required sections are present")
        return True


def validate_subsections(content):
    """Validate presence of important subsections"""
    print("\n" + "="*50)
    print("SUBSECTION VALIDATION")
    print("="*50)
    
    important_subsections = [
        "Problem Definition",
        "Practical Applications", 
        "Project Objectives",
        "Dataset Description",
        "Class Distribution",
        "Feature Engineering",
        "Pipeline Overview",
        "Performance Comparison",
        "Feature Importance", 
        "Model Performance Ranking",
        "Key Findings",
        "Future Work"
    ]
    
    found_subsections = []
    
    for subsection in important_subsections:
        pattern = rf'#{2,4}\s+.*{re.escape(subsection)}'
        if re.search(pattern, content, re.IGNORECASE):
            print(f"✓ Subsection '{subsection}' found")
            found_subsections.append(subsection)
        else:
            print(f"⚠️  Subsection '{subsection}' not found")
    
    print(f"\nSubsection summary: {len(found_subsections)}/{len(important_subsections)} subsections found")
    return len(found_subsections) >= len(important_subsections) * 0.8  # 80% threshold


def validate_tables(content):
    """Validate presence and formatting of tables"""
    print("\n" + "="*50)
    print("TABLE VALIDATION")
    print("="*50)
    
    # Find markdown tables
    table_pattern = r'\|.*\|.*\n\|[-\s\|:]+\|.*\n(\|.*\|.*\n)+'
    tables = re.findall(table_pattern, content)
    
    print(f"Found {len(tables)} tables in the document")
    
    # Look for specific important tables
    important_tables = [
        "Class Distribution",
        "Model Performance", 
        "Feature Importance",
        "Hyperparameter"
    ]
    
    found_tables = []
    for table_name in important_tables:
        if re.search(f"Table.*{table_name}", content, re.IGNORECASE):
            print(f"✓ Table '{table_name}' found")
            found_tables.append(table_name)
        else:
            print(f"⚠️  Table '{table_name}' not found")
    
    print(f"\nImportant tables: {len(found_tables)}/{len(important_tables)} found")
    return len(tables) >= 3  # At least 3 tables expected


def validate_figures(content):
    """Validate figure references and file existence"""
    print("\n" + "="*50)
    print("FIGURE VALIDATION")
    print("="*50)
    
    # Find all figure references
    figure_refs = re.findall(r'!\[.*?\]\((.*?)\)', content)
    print(f"Found {len(figure_refs)} figure references")
    
    if not figure_refs:
        print("⚠️  No figure references found in document")
        return False
    
    # Check if referenced files exist
    missing_figures = []
    existing_figures = []
    
    for fig_ref in figure_refs:
        if os.path.exists(fig_ref):
            print(f"✓ Figure exists: {fig_ref}")
            existing_figures.append(fig_ref)
        else:
            print(f"❌ Figure missing: {fig_ref}")
            missing_figures.append(fig_ref)
    
    print(f"\nFigure summary: {len(existing_figures)}/{len(figure_refs)} figures exist")
    
    if missing_figures:
        print("\nMissing figures:")
        for fig in missing_figures:
            print(f"  - {fig}")
        return False
    else:
        print("✅ All referenced figures exist")
        return True


def validate_references(content):
    """Validate references section"""
    print("\n" + "="*50)
    print("REFERENCES VALIDATION")
    print("="*50)
    
    # Find references section
    ref_match = re.search(r'## References\s+([\s\S]+?)(?=##|$)', content)
    if not ref_match:
        print("❌ References section not found!")
        return False
    
    ref_content = ref_match.group(1)
    
    # Count numbered references
    numbered_refs = re.findall(r'^\d+\.', ref_content, re.MULTILINE)
    print(f"Found {len(numbered_refs)} numbered references")
    
    # Check for key references
    key_references = [
        "Dataset Source",
        "Scikit-learn",
        "LightGBM", 
        "GPyTorch",
        "SHAP"
    ]
    
    found_refs = []
    for ref in key_references:
        if re.search(ref, ref_content, re.IGNORECASE):
            print(f"✓ Reference '{ref}' found")
            found_refs.append(ref)
        else:
            print(f"⚠️  Reference '{ref}' not found")
    
    print(f"\nKey references: {len(found_refs)}/{len(key_references)} found")
    return len(numbered_refs) >= 5 and len(found_refs) >= 3


def validate_content_quality(content):
    """Validate content quality indicators"""
    print("\n" + "="*50)
    print("CONTENT QUALITY VALIDATION")
    print("="*50)
    
    # Check document length
    word_count = len(content.split())
    print(f"Total document word count: {word_count}")
    
    if word_count < 3000:
        print("⚠️  Document seems short for an academic report")
    elif word_count > 8000:
        print("⚠️  Document seems very long for this assignment")
    else:
        print("✓ Document length is appropriate")
    
    # Check for code blocks
    code_blocks = len(re.findall(r'```.*?```', content, re.DOTALL))
    print(f"Code blocks found: {code_blocks}")
    
    # Check for mathematical notation
    math_notation = len(re.findall(r'\$.*?\$', content))
    print(f"Mathematical expressions found: {math_notation}")
    
    # Check for emphasis (bold/italic)
    bold_text = len(re.findall(r'\*\*.*?\*\*', content))
    italic_text = len(re.findall(r'\*.*?\*', content))
    print(f"Bold text instances: {bold_text}")
    print(f"Italic text instances: {italic_text}")
    
    # Check for proper academic tone indicators
    academic_phrases = [
        "This project", "The results", "Our analysis", "The findings",
        "In conclusion", "Furthermore", "However", "Therefore",
        "demonstrates", "reveals", "indicates", "suggests"
    ]
    
    found_phrases = []
    for phrase in academic_phrases:
        if re.search(phrase, content, re.IGNORECASE):
            found_phrases.append(phrase)
    
    print(f"Academic tone indicators: {len(found_phrases)}/{len(academic_phrases)} found")
    
    return word_count >= 3000 and len(found_phrases) >= len(academic_phrases) * 0.5


def generate_validation_report(results):
    """Generate final validation report"""
    print("\n" + "="*60)
    print("FINAL VALIDATION REPORT")
    print("="*60)
    
    total_checks = len(results)
    passed_checks = sum(results.values())
    
    print(f"Validation Summary: {passed_checks}/{total_checks} checks passed")
    print()
    
    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {check_name}")
    
    print("\n" + "="*60)
    
    if passed_checks == total_checks:
        print("🎉 DOCUMENT VALIDATION SUCCESSFUL!")
        print("Your DOCUMENT.md meets all quality standards.")
        return True
    elif passed_checks >= total_checks * 0.8:
        print("⚠️  DOCUMENT VALIDATION MOSTLY SUCCESSFUL")
        print("Minor issues found. Please review the failed checks above.")
        return True
    else:
        print("❌ DOCUMENT VALIDATION FAILED")
        print("Significant issues found. Please address the failed checks.")
        return False


def main():
    """Main validation function"""
    document_path = "DOCUMENT.md"
    
    print("COMP4702 Assignment Document Validator")
    print("="*60)
    print(f"Validating: {document_path}")
    
    if not os.path.exists(document_path):
        print(f"❌ Document file not found: {document_path}")
        sys.exit(1)
    
    # Read document content
    try:
        with open(document_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading document: {e}")
        sys.exit(1)
    
    print(f"✓ Document loaded successfully ({len(content)} characters)")
    
    # Run all validation checks
    validation_results = {
        "Abstract Format": validate_abstract(content),
        "Section Structure": validate_sections(content), 
        "Subsection Completeness": validate_subsections(content),
        "Table Presence": validate_tables(content),
        "Figure References": validate_figures(content),
        "References Quality": validate_references(content),
        "Content Quality": validate_content_quality(content)
    }
    
    # Generate final report
    success = generate_validation_report(validation_results)
    
    if success:
        print("\n✅ Document validation completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Document validation failed. Please review and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main() 