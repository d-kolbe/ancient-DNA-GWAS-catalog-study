#!/usr/bin/env python3
"""
Category utilities for exact-match GWAS categories
Much simpler than regex-based version
"""

import yaml

def load_categories(yaml_file):
    """
    Load categories from YAML file
    
    Returns:
        dict: {category_name: [list of trait names]}
    """
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)
        if 'categories' in data:
            return data['categories']
        else:
            return data

def sanitize_category_name(name):
    """
    Sanitize category name for use in filenames
    Removes spaces, special characters that could break SLURM/shell
    """
    # Replace common separators
    name = (name.replace(' - ', '_-_')
                .replace(' & ', '_and_')
                .replace(' ', '_')
                .replace('/', '-')
                .replace('(', '')
                .replace(')', ''))
    
    # Remove all types of apostrophes/quotes
    for apostrophe in ["'", "'", "'", "`", "´", "'"]:
        name = name.replace(apostrophe, '')
    
    # Remove other shell-breaking characters
    for char in ['"', '"', '"', '$', '!', ';', '|', '<', '>', '&', '*', '?', '[', ']', '{', '}']:
        name = name.replace(char, '')
    
    return name

def get_all_categories(yaml_file):
    """
    Get all category names and their sanitized versions
    
    Returns:
        dict: {sanitized_name: {'full_name': str, 'traits': [list]}}
    """
    categories = load_categories(yaml_file)
    
    result = {}
    for cat_name, traits in categories.items():
        safe_name = sanitize_category_name(cat_name)
        result[safe_name] = {
            'full_name': cat_name,
            'traits': traits
        }
    
    return result