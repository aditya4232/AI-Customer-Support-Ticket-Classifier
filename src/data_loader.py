"""
Data loader module for loading and processing ticket data from CSV.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from pathlib import Path


class TicketDataLoader:
    """Load and process ticket data from CSV database."""
    
    def __init__(self, csv_path: str = "data/ticket_database.csv"):
        """
        Initialize the data loader.
        
        Args:
            csv_path: Path to the CSV database file
        """
        self.csv_path = csv_path
        self.data = None
        self.industries = None
        self.categories = None
        self.priorities = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Returns:
            DataFrame containing ticket data
        """
        try:
            self.data = pd.read_csv(self.csv_path)
            self._extract_categories_and_priorities()
            return self.data
        except FileNotFoundError:
            raise FileNotFoundError(f"Database file not found: {self.csv_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def _extract_categories_and_priorities(self):
        """Extract unique industries, categories, and priorities from data."""
        if self.data is not None:
            self.industries = self.data['industry'].unique().tolist()
            self.categories = self.data['category'].unique().tolist()
            self.priorities = self.data['priority'].unique().tolist()
    
    def get_train_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get all training data.
        
        Returns:
            Tuple of (texts, industries, categories, priorities)
        """
        if self.data is None:
            self.load_data()
        
        return (
            self.data['ticket_text'].values,
            self.data['industry'].values,
            self.data['category'].values,
            self.data['priority'].values
        )
    
    def get_industry_distribution(self) -> Dict[str, int]:
        """
        Get distribution of industries in the dataset.
        
        Returns:
            Dictionary mapping industry to count
        """
        if self.data is None:
            self.load_data()
        
        return self.data['industry'].value_counts().to_dict()
    
    def get_category_distribution(self) -> Dict[str, int]:
        """
        Get distribution of categories in the dataset.
        
        Returns:
            Dictionary mapping category to count
        """
        if self.data is None:
            self.load_data()
        
        return self.data['category'].value_counts().to_dict()
    
    def get_priority_distribution(self) -> Dict[str, int]:
        """
        Get distribution of priorities in the dataset.
        
        Returns:
            Dictionary mapping priority to count
        """
        if self.data is None:
            self.load_data()
        
        return self.data['priority'].value_counts().to_dict()
    
    def get_info(self) -> Dict:
        """
        Get summary information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        if self.data is None:
            self.load_data()
        
        return {
            'total_samples': len(self.data),
            'industries': self.industries,
            'categories': self.categories,
            'priorities': self.priorities,
            'industry_distribution': self.get_industry_distribution(),
            'category_distribution': self.get_category_distribution(),
            'priority_distribution': self.get_priority_distribution()
        }


def load_ticket_data(csv_path: str = "data/ticket_database.csv") -> pd.DataFrame:
    """
    Convenience function to load ticket data.
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        DataFrame with ticket data
    """
    loader = TicketDataLoader(csv_path)
    return loader.load_data()


if __name__ == "__main__":
    # Test the data loader
    loader = TicketDataLoader()
    info = loader.get_info()
    
    print("Dataset Information:")
    print("-" * 50)
    print(f"Total samples: {info['total_samples']}")
    print(f"\nIndustries ({len(info['industries'])}):")
    for ind, count in info['industry_distribution'].items():
        print(f"  - {ind}: {count}")
    print(f"\nCategories ({len(info['categories'])}):")
    for cat, count in info['category_distribution'].items():
        print(f"  - {cat}: {count}")
    print(f"\nPriorities ({len(info['priorities'])}):")
    for pri, count in info['priority_distribution'].items():
        print(f"  - {pri}: {count}")
