import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('viridis')

# Function to load and combine data
def load_data(file_paths):
    """
    Load and combine multiple CSV files into a single DataFrame
    """
    dataframes = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df

# Function to clean the data
def clean_data(df):
    """
    Clean the data by handling missing values, converting data types, etc.
    """
    # Convert date columns to datetime
    for col in ['open_dt', 'closed_dt']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Filter for animal-related complaints
    animal_complaints = df[df['case_title'].str.contains('Animal|Dog|Cat|Rat|Wildlife|Bird', 
                                                         case=False, na=False)]
    
    # Handle missing lat/long
    animal_complaints = animal_complaints.dropna(subset=['latitude', 'longitude'])
    
    # Create year and month columns for time analysis
    animal_complaints['year'] = animal_complaints['open_dt'].dt.year
    animal_complaints['month'] = animal_complaints['open_dt'].dt.month
    animal_complaints['day_of_week'] = animal_complaints['open_dt'].dt.day_name()
    animal_complaints['quarter'] = animal_complaints['open_dt'].dt.quarter
    
    return animal_complaints

# Function to analyze trends over time
def analyze_time_trends(df):
    """
    Analyze trends in complaints over time (yearly, monthly, etc.)
    """
    # Yearly trends
    yearly_counts = df.groupby('year').size()
    plt.figure(figsize=(10, 6))
    yearly_counts.plot(kind='bar', color='steelblue')
    plt.title('Animal Complaints by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('../outputs/yearly_complaints.png')
    
    # Monthly trends
    monthly_counts = df.groupby(['year', 'month']).size().unstack()
    plt.figure(figsize=(12, 6))
    monthly_counts.plot(kind='line', marker='o')
    plt.title('Monthly Animal Complaints by Year')
    plt.xlabel('Month')
    plt.ylabel('Number of Complaints')
    plt.xticks(range(1, 13))
    plt.legend(title='Year')
    plt.tight_layout()
    plt.savefig('../outputs/monthly_complaints.png')
    
    # Day of week analysis
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_counts = df['day_of_week'].value_counts().reindex(day_order)
    plt.figure(figsize=(10, 6))
    day_counts.plot(kind='bar', color='steelblue')
    plt.title('Animal Complaints by Day of Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('../outputs/day_of_week_complaints.png')
    
    return {
        'yearly_counts': yearly_counts,
        'monthly_counts': monthly_counts,
        'day_counts': day_counts
    }

# Function to analyze animal types
def analyze_animal_types(df):
    """
    Analyze the types of animals reported in complaints
    """
    # Extract animal type from case_title or subject
    def extract_animal_type(text):
        if pd.isna(text):
            return 'Unknown'
        
        text = text.lower()
        if 'rat' in text or 'rodent' in text:
            return 'Rat/Rodent'
        elif 'dog' in text:
            return 'Dog'
        elif 'cat' in text:
            return 'Cat'
        elif 'raccoon' in text:
            return 'Raccoon'
        elif 'bird' in text or 'pigeon' in text:
            return 'Bird'
        elif 'squirrel' in text:
            return 'Squirrel'
        elif 'coyote' in text:
            return 'Coyote'
        elif 'animal' in text or 'wildlife' in text:
            return 'Other Animal'
        else:
            return 'Unknown'
    
    df['animal_type'] = df['case_title'].apply(extract_animal_type)
    df.loc[df['animal_type'] == 'Unknown', 'animal_type'] = df['subject'].apply(extract_animal_type)
    
    # Count complaints by animal type
    animal_counts = df['animal_type'].value_counts()
    
    plt.figure(figsize=(10, 6))
    animal_counts.plot(kind='bar', color='darkgreen')
    plt.title('Complaints by Animal Type')
    plt.xlabel('Animal Type')
    plt.ylabel('Number of Complaints')
    plt.tight_layout()
    plt.savefig('../outputs/animal_type_complaints.png')
    
    return animal_counts

# Function to analyze geographic patterns
def analyze_geographic_patterns(df, num_clusters=5):
    """
    Analyze geographic patterns in animal complaints using clustering
    """
    # Extract coordinates for clustering
    coords = df[['latitude', 'longitude']].copy()
    
    # Standardize the coordinates
    scaler = StandardScaler()
    scaled_coords = scaler.fit_transform(coords)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(scaled_coords)
    
    # Plot the clusters
    plt.figure(figsize=(12, 10))
    
    # Create a scatter plot with cluster colors
    scatter = plt.scatter(df['longitude'], df['latitude'], 
                         c=df['cluster'], cmap='viridis', 
                         alpha=0.6, s=10)
    
    # Add cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 1], centers[:, 0], c='red', s=100, alpha=0.8, marker='X')
    
    plt.title('Geographic Clusters of Animal Complaints')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.colorbar(scatter, label='Cluster')
    plt.tight_layout()
    plt.savefig('../outputs/geographic_clusters.png')
    
    # Analyze animal types within each cluster
    cluster_animal_counts = df.groupby(['cluster', 'animal_type']).size().unstack().fillna(0)
    
    # Plot animal distribution by cluster
    plt.figure(figsize=(14, 8))
    cluster_animal_counts.plot(kind='bar', stacked=True)
    plt.title('Animal Types by Geographic Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Number of Complaints')
    plt.legend(title='Animal Type')
    plt.tight_layout()
    plt.savefig('../outputs/animal_types_by_cluster.png')
    
    return {
        'cluster_centers': centers,
        'cluster_animal_counts': cluster_animal_counts
    }

# Function to analyze seasonal patterns
def analyze_seasonal_patterns(df):
    """
    Analyze seasonal patterns in animal complaints
    """
    # Analyze by quarter
    quarter_counts = df.groupby(['year', 'quarter']).size().unstack()
    
    plt.figure(figsize=(12, 6))
    quarter_counts.plot(kind='bar', stacked=True)
    plt.title('Animal Complaints by Quarter and Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Complaints')
    plt.legend(title='Quarter')
    plt.tight_layout()
    plt.savefig('../outputs/quarterly_complaints.png')
    
    # Analyze by month across all years
    monthly_all_years = df.groupby('month').size()
    
    plt.figure(figsize=(10, 6))
    monthly_all_years.plot(kind='line', marker='o', color='purple')
    plt.title('Seasonal Pattern of Animal Complaints (All Years)')
    plt.xlabel('Month')
    plt.ylabel('Number of Complaints')
    plt.xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig('../outputs/seasonal_pattern.png')
    
    # Analyze animal types by season
    df['season'] = pd.cut(
        df['month'], 
        bins=[0, 3, 6, 9, 12], 
        labels=['Winter', 'Spring', 'Summer', 'Fall'],
        include_lowest=True
    )
    
    season_animal_counts = df.groupby(['season', 'animal_type']).size().unstack().fillna(0)
    
    plt.figure(figsize=(12, 6))
    season_animal_counts.plot(kind='bar', stacked=True)
    plt.title('Animal Types by Season')
    plt.xlabel('Season')
    plt.ylabel('Number of Complaints')
    plt.legend(title='Animal Type')
    plt.tight_layout()
    plt.savefig('../outputs/animal_types_by_season.png')
    
    return {
        'quarter_counts': quarter_counts,
        'monthly_all_years': monthly_all_years,
        'season_animal_counts': season_animal_counts
    }

# Function to generate a comprehensive report
def generate_report(time_trends, animal_types, geo_patterns, seasonal_patterns):
    """
    Generate a comprehensive report of findings
    """
    report = "# Animal Complaints Analysis Report\n\n"
    
    # Time trends
    report += "## Time Trends Analysis\n\n"
    report += f"- Total complaints analyzed: {sum(time_trends['yearly_counts'])}\n"
    report += f"- Year with most complaints: {time_trends['yearly_counts'].idxmax()} ({time_trends['yearly_counts'].max()} complaints)\n"
    report += f"- Year with fewest complaints: {time_trends['yearly_counts'].idxmin()} ({time_trends['yearly_counts'].min()} complaints)\n\n"
    
    # Animal types
    report += "## Animal Type Analysis\n\n"
    report += "Top 5 most reported animal types:\n\n"
    for i, (animal, count) in enumerate(animal_types.head(5).items(), 1):
        report += f"{i}. {animal}: {count} complaints ({count/sum(animal_types)*100:.1f}%)\n"
    report += "\n"
    
    # Geographic patterns
    report += "## Geographic Pattern Analysis\n\n"
    report += f"- Number of geographic clusters identified: {len(geo_patterns['cluster_centers'])}\n"
    report += "- Most common animal type in each cluster:\n\n"
    
    for cluster in geo_patterns['cluster_animal_counts'].index:
        most_common = geo_patterns['cluster_animal_counts'].loc[cluster].idxmax()
        count = geo_patterns['cluster_animal_counts'].loc[cluster, most_common]
        report += f"  - Cluster {cluster}: {most_common} ({count} complaints)\n"
    report += "\n"
    
    # Seasonal patterns
    report += "## Seasonal Pattern Analysis\n\n"
    report += "- Season with most complaints: "
    max_season = seasonal_patterns['season_animal_counts'].sum(axis=1).idxmax()
    max_season_count = seasonal_patterns['season_animal_counts'].sum(axis=1).max()
    report += f"{max_season} ({max_season_count} complaints)\n"
    
    report += "- Most common animal type by season:\n\n"
    for season in seasonal_patterns['season_animal_counts'].index:
        most_common = seasonal_patterns['season_animal_counts'].loc[season].idxmax()
        count = seasonal_patterns['season_animal_counts'].loc[season, most_common]
        report += f"  - {season}: {most_common} ({count} complaints)\n"
    report += "\n"
    
    # Conclusions
    report += "## Key Findings and Recommendations\n\n"
    report += "1. **Temporal Trends**: "
    if time_trends['yearly_counts'].iloc[-1] > time_trends['yearly_counts'].iloc[0]:
        report += "Animal complaints have increased over the years, suggesting a growing concern.\n"
    else:
        report += "Animal complaints have decreased over the years, suggesting effective city interventions.\n"
    
    report += "2. **Animal Type Focus**: "
    top_animal = animal_types.idxmax()
    report += f"Focus resources on addressing {top_animal} issues as they represent the majority of complaints.\n"
    
    report += "3. **Geographic Prioritization**: Target interventions in clusters with high complaint volumes, particularly for specific animal types prevalent in those areas.\n"
    
    report += "4. **Seasonal Planning**: "
    report += f"Allocate additional resources during {max_season} when complaint volumes peak, especially for {seasonal_patterns['season_animal_counts'].loc[max_season].idxmax()} control.\n"
    
    report += "\n*Report generated on " + datetime.now().strftime("%Y-%m-%d") + "*"
    
    # Save report to file
    with open('../outputs/analysis_report.md', 'w') as f:
        f.write(report)
    
    return report

def main():
    """
    Main function to execute the analysis pipeline
    """
    # Create outputs directory if it doesn't exist
    if not os.path.exists('../outputs'):
        os.makedirs('../outputs')
    
    # Load data from all CSV files
    file_paths = ['../2015.csv', '../2016.csv', '../2017.csv', '../2018.csv', '../2019.csv']
    print("Loading data from CSV files...")
    combined_df = load_data(file_paths)
    print(f"Loaded {len(combined_df)} total records.")
    
    # Clean the data
    print("Cleaning data and filtering for animal complaints...")
    animal_df = clean_data(combined_df)
    print(f"Found {len(animal_df)} animal-related complaints.")
    
    # Save cleaned data
    animal_df.to_csv('../outputs/cleaned_animal_complaints.csv', index=False)
    print("Saved cleaned data to '../outputs/cleaned_animal_complaints.csv'")
    
    # Analyze time trends
    print("Analyzing time trends...")
    time_trends = analyze_time_trends(animal_df)
    
    # Analyze animal types
    print("Analyzing animal types...")
    animal_types = analyze_animal_types(animal_df)
    
    # Analyze geographic patterns
    print("Analyzing geographic patterns...")
    geo_patterns = analyze_geographic_patterns(animal_df)
    
    # Analyze seasonal patterns
    print("Analyzing seasonal patterns...")
    seasonal_patterns = analyze_seasonal_patterns(animal_df)
    
    # Generate report
    print("Generating comprehensive report...")
    generate_report(time_trends, animal_types, geo_patterns, seasonal_patterns)
    print("Analysis complete! Report saved to '../outputs/analysis_report.md'")
    
    return animal_df

if __name__ == "__main__":
    main() 