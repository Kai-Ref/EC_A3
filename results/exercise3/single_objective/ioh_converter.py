import pickle
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd


class IOHAnalyzerConverter:
    """
    Convert benchmark results from single_objective.py to IOHanalyzer format.
    
    Data structure in pickle file:
    all_results[problem_id][pop_size][mechanism] = {
        'best_fitness': [final fitness per run],
        'best_f_vals': [fitness progression per run],
        'diversity': [diversity history per run]
    }
    """
    
    def __init__(self, pickle_file, output_dir='ioh_data'):
        """
        Initialize converter.
        
        Args:
            pickle_file: Path to all_results.pkl file
            output_dir: Directory to save IOHanalyzer formatted data
        """
        self.pickle_file = pickle_file
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load results
        with open(pickle_file, 'rb') as f:
            self.results = pickle.load(f)
        
        print(f"Loaded results from: {pickle_file}")
        self._print_summary()
    
    def _get_improvements(self, fitness_progression, maximization):
        """
        Extract only evaluations where fitness improves (target-based tracking).
        
        Args:
            fitness_progression: List of fitness values over evaluations
            maximization: True if maximization, False if minimization
        
        Returns:
            List of (evaluation_number, fitness_value) tuples for improvements
        """
        if not fitness_progression:
            return []
        
        improvements = []
        current_best = fitness_progression[0]
        improvements.append((1, current_best))  # Always include first evaluation
        
        for eval_count, fitness_value in enumerate(fitness_progression[1:], start=2):
            # Check if this is an improvement
            if maximization:
                if fitness_value > current_best:
                    current_best = fitness_value
                    improvements.append((eval_count, fitness_value))
            else:
                if fitness_value < current_best:
                    current_best = fitness_value
                    improvements.append((eval_count, fitness_value))
        
        # CRITICAL: Always include the final evaluation (even if no improvement)
        # This is required for correct ERT calculation in IOHanalyzer
        final_eval = len(fitness_progression)
        final_fitness = fitness_progression[-1]
        if improvements[-1][0] != final_eval:
            improvements.append((final_eval, final_fitness))
        
        return improvements
    
    def _print_summary(self):
        """Print summary of loaded data."""
        problems = list(self.results.keys())
        print(f"\nData Summary:")
        print(f"  Problems: {problems}")
        
        if problems:
            first_problem = problems[0]
            pop_sizes = list(self.results[first_problem].keys())
            print(f"  Population sizes: {pop_sizes}")
            
            if pop_sizes:
                first_pop = pop_sizes[0]
                mechanisms = list(self.results[first_problem][first_pop].keys())
                print(f"  Mechanisms: {mechanisms}")
                
                if mechanisms:
                    first_mech = mechanisms[0]
                    n_runs = len(self.results[first_problem][first_pop][first_mech]['best_fitness'])
                    print(f"  Runs per configuration: {n_runs}")
    
    def to_csv(self, csv_filename='benchmark_results.csv', maximization=True, 
               downsample=True):
        """
        Convert to simple CSV format (easiest option for IOHanalyzer).
        
        Args:
            csv_filename: Name of output CSV file
            maximization: True if maximization problem, False if minimization
            downsample: If True, only record evaluations where fitness improves
        
        Returns:
            DataFrame with the converted data
        """
        data_rows = []
        
        print("\nConverting to CSV format...")
        if downsample:
            print("  Using downsampled format (only improvements)")
        
        for problem_id, problem_data in self.results.items():
            for pop_size, pop_data in problem_data.items():
                for mechanism, mech_data in pop_data.items():
                    best_f_vals_list = mech_data['best_f_vals']
                    
                    # Process each run
                    for run_id, fitness_progression in enumerate(best_f_vals_list):
                        # Skip empty runs
                        if not fitness_progression:
                            continue
                        
                        # Create algorithm ID combining pop_size and mechanism
                        algorithm_id = f"{mechanism}_pop{pop_size}"
                        
                        if downsample:
                            # Only record evaluations where fitness improves
                            improvements = self._get_improvements(fitness_progression, maximization)
                            for eval_count, fitness_value in improvements:
                                # Skip infinite values
                                if not np.isfinite(fitness_value):
                                    continue
                                    
                                data_rows.append({
                                    'evaluations': int(eval_count),
                                    'raw_y': float(fitness_value),
                                    'function_id': int(problem_id),
                                    'algorithm_id': str(algorithm_id),
                                    'dimension': int(pop_size),
                                    'run_id': int(run_id + 1),
                                    'instance': int(1)
                                })
                        else:
                            # Record all evaluations
                            for eval_count, fitness_value in enumerate(fitness_progression, start=1):
                                # Skip infinite values
                                if not np.isfinite(fitness_value):
                                    continue
                                    
                                data_rows.append({
                                    'evaluations': int(eval_count),
                                    'raw_y': float(fitness_value),
                                    'function_id': int(problem_id),
                                    'algorithm_id': str(algorithm_id),
                                    'dimension': int(pop_size),
                                    'run_id': int(run_id + 1),
                                    'instance': int(1)
                                })
        
        # Create DataFrame and save
        df = pd.DataFrame(data_rows)
        
        # Ensure proper data types
        df['evaluations'] = df['evaluations'].astype(int)
        df['raw_y'] = df['raw_y'].astype(float)
        df['function_id'] = df['function_id'].astype(int)
        df['dimension'] = df['dimension'].astype(int)
        df['run_id'] = df['run_id'].astype(int)
        df['instance'] = df['instance'].astype(int)
        
        # Sort by function, algorithm, dimension, run, and evaluation
        df = df.sort_values(['function_id', 'algorithm_id', 'dimension', 'run_id', 'evaluations'])
        
        csv_path = self.output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ CSV file saved to: {csv_path}")
        print(f"  Total data points: {len(df)}")
        print(f"  Unique runs: {df.groupby(['function_id', 'algorithm_id', 'run_id']).ngroups}")
        print(f"\n{'='*70}")
        print("To use in IOHanalyzer:")
        print("="*70)
        print("1. Go to IOHanalyzer web interface")
        print("2. Enable 'use custom csv format' checkbox")
        print(f"3. Upload {csv_filename}")
        print("4. Map columns as follows:")
        print("   - Evaluation counter: evaluations")
        print("   - Function values: raw_y") 
        print("   - Function ID: function_id")
        print("   - Algorithm ID: algorithm_id")
        print("   - Problem dimension: dimension")
        print("   - Run ID: run_id")
        print(f"5. {'Check' if maximization else 'Uncheck'} the Maximization checkbox")
        print(f"6. Click 'Process'")
        print("="*70)
        
        # Also save a simpler version for troubleshooting
        simple_csv = csv_filename.replace('.csv', '_simple.csv')
        simple_path = self.output_dir / simple_csv
        df_simple = df[['evaluations', 'raw_y']].copy()
        df_simple.to_csv(simple_path, index=False)
        print(f"\nAlso created simplified version: {simple_csv}")
        print("(Use this if you want to manually enter function/algorithm IDs)")
        
        return df
    
    def to_json(self, maximization=True, suite_name='SubmodularOptimization'):
        """
        Convert to JSON format (IOHexperimenter 0.3.3+).
        Creates separate JSON files and .dat files for each function.
        
        Args:
            maximization: True if maximization, False if minimization
            suite_name: Name of the benchmark suite
        """
        print("\nConverting to JSON format (IOHexperimenter 0.3.3+)...")
        
        # Process each problem (function)
        for problem_id, problem_data in self.results.items():
            print(f"\nProcessing Problem {problem_id}...")
            
            # Create data directory for this function
            func_dir = self.output_dir / f'data_f{problem_id}'
            func_dir.mkdir(exist_ok=True)
            
            scenarios = []
            
            # Group by algorithm (mechanism + pop_size combination)
            for pop_size, pop_data in problem_data.items():
                for mechanism, mech_data in pop_data.items():
                    algorithm_name = f"{mechanism}_pop{pop_size}"
                    
                    # Use pop_size as dimension (or you could use actual problem dimension)
                    dimension = pop_size
                    
                    # Create .dat file for this configuration
                    dat_filename = f'IOHprofiler_f{problem_id}_{algorithm_name}_DIM{dimension}.dat'
                    dat_path = func_dir / dat_filename
                    
                    # Prepare runs data
                    runs = []
                    best_f_vals_list = mech_data['best_f_vals']
                    
                    for run_id, fitness_progression in enumerate(best_f_vals_list, start=1):
                        if not fitness_progression:
                            continue
                        
                        # Get best value and its evaluation
                        if maximization:
                            best_value = max(fitness_progression)
                            best_eval = fitness_progression.index(best_value) + 1
                        else:
                            best_value = min(fitness_progression)
                            best_eval = fitness_progression.index(best_value) + 1
                        
                        total_evals = len(fitness_progression)
                        
                        run_info = {
                            'instance': 1,
                            'evals': int(total_evals),
                            'best': {
                                'evals': int(best_eval),
                                'y': float(best_value)
                            }
                        }
                        runs.append(run_info)
                    
                    # Write .dat file with full trajectory
                    self._write_dat_file(dat_path, best_f_vals_list, maximization)
                    
                    # Add scenario
                    scenario = {
                        'dimension': int(dimension),
                        'path': f'{func_dir.name}/{dat_filename}',
                        'runs': runs
                    }
                    scenarios.append(scenario)
            
            # Create JSON metadata file for this function
            # Note: IOHanalyzer expects one algorithm per JSON file
            # We'll create separate JSON files for each algorithm
            algorithm_scenarios = {}
            for scenario in scenarios:
                # Extract algorithm name from path
                dat_file = scenario['path'].split('/')[-1]
                alg_name = '_'.join(dat_file.split('_')[2:-1])  # Extract algorithm name
                
                if alg_name not in algorithm_scenarios:
                    algorithm_scenarios[alg_name] = []
                algorithm_scenarios[alg_name].append(scenario)
            
            # Create a JSON file for each algorithm
            for alg_name, alg_scenarios in algorithm_scenarios.items():
                metadata = {
                    'version': '0.3.3',
                    'suite': suite_name,
                    'function_id': int(problem_id),
                    'function_name': f'Problem{problem_id}',
                    'maximization': maximization,
                    'algorithm': {
                        'name': alg_name,
                        'info': f'Algorithm configuration: {alg_name}'
                    },
                    'attributes': ['evaluations', 'raw_y'],
                    'scenarios': alg_scenarios
                }
                
                json_path = self.output_dir / f'IOHprofiler_f{problem_id}_{alg_name}.info'
                with open(json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"  ✓ Created: {json_path.name}")
        
        print(f"\n✓ JSON format conversion complete!")
        print(f"  Output directory: {self.output_dir}")
    
    def _write_dat_file(self, filepath, fitness_progressions, maximization):
        """
        Write raw data .dat file with downsampled trajectories (only improvements).
        
        Args:
            filepath: Path to save the .dat file
            fitness_progressions: List of fitness progressions (one per run)
            maximization: True if maximization, False if minimization
        """
        with open(filepath, 'w') as f:
            # Write header
            f.write('evaluations raw_y\n')
            
            # Write each run
            for run_progression in fitness_progressions:
                if not run_progression:
                    continue
                
                # Get only improvements for this run
                improvements = self._get_improvements(run_progression, maximization)
                
                # Write trajectory for this run
                for eval_count, fitness_value in improvements:
                    f.write(f'{eval_count} {fitness_value:.10e}\n')
                
                # Separator line between runs (new format uses blank line)
                f.write('\n')
    
    def create_combined_csv(self, csv_filename='combined_results.csv', maximization=True,
                           downsample=True):
        """
        Create a more detailed CSV that includes diversity metrics.
        
        Args:
            csv_filename: Name of output CSV file
            maximization: True if maximization problem, False if minimization
            downsample: If True, only record evaluations where fitness improves
        """
        data_rows = []
        
        print("\nCreating combined CSV with diversity metrics...")
        if downsample:
            print("  Using downsampled format (only improvements)")
        
        for problem_id, problem_data in self.results.items():
            for pop_size, pop_data in problem_data.items():
                for mechanism, mech_data in pop_data.items():
                    best_f_vals_list = mech_data['best_f_vals']
                    diversity_list = mech_data.get('diversity', [])
                    
                    for run_id, fitness_progression in enumerate(best_f_vals_list):
                        algorithm_id = f"{mechanism}_pop{pop_size}"
                        
                        # Get diversity for this run if available
                        diversity_progression = diversity_list[run_id] if run_id < len(diversity_list) else []
                        
                        if downsample:
                            improvements = self._get_improvements(fitness_progression, maximization)
                            
                            for eval_count, fitness_value in improvements:
                                # Find corresponding diversity value
                                div_idx = eval_count // 100
                                diversity_value = diversity_progression[div_idx] if div_idx < len(diversity_progression) else None
                                
                                row = {
                                    'evaluations': eval_count,
                                    'raw_y': fitness_value,
                                    'function_id': problem_id,
                                    'algorithm_id': algorithm_id,
                                    'dimension': pop_size,
                                    'run_id': run_id + 1,
                                    'instance': 1,
                                    'mechanism': mechanism,
                                    'pop_size': pop_size
                                }
                                
                                if diversity_value is not None:
                                    row['diversity'] = diversity_value
                                
                                data_rows.append(row)
                        else:
                            for eval_count, fitness_value in enumerate(fitness_progression, start=1):
                                # Find corresponding diversity value
                                div_idx = eval_count // 100
                                diversity_value = diversity_progression[div_idx] if div_idx < len(diversity_progression) else None
                                
                                row = {
                                    'evaluations': eval_count,
                                    'raw_y': fitness_value,
                                    'function_id': problem_id,
                                    'algorithm_id': algorithm_id,
                                    'dimension': pop_size,
                                    'run_id': run_id + 1,
                                    'instance': 1,
                                    'mechanism': mechanism,
                                    'pop_size': pop_size
                                }
                                
                                if diversity_value is not None:
                                    row['diversity'] = diversity_value
                                
                                data_rows.append(row)
        
        df = pd.DataFrame(data_rows)
        csv_path = self.output_dir / csv_filename
        df.to_csv(csv_path, index=False)
        
        print(f"\n✓ Combined CSV saved to: {csv_path}")
        return df
    
    def to_iohprofiler_format(self, maximization=True):
        """
        Convert to standard IOHprofiler .dat and .info format (most reliable).
        This creates the directory structure that IOHanalyzer expects.
        Uses the LEGACY format (IOHexperimenter 0.3.2 and below) which is most stable.
        
        Args:
            maximization: True if maximization, False if minimization
        """
        print("\nConverting to IOHprofiler standard format (legacy)...")
        
        # Process each problem (function)
        for problem_id, problem_data in self.results.items():
            print(f"\nProcessing Problem {problem_id}...")
            
            # Group all configurations for this problem
            # Each algorithm gets its own .info file, but shares the function ID
            for pop_size, pop_data in problem_data.items():
                for mechanism, mech_data in pop_data.items():
                    algorithm_name = f"{mechanism}_pop{pop_size}"
                    
                    # Create data directory for this function
                    func_dir = self.output_dir / f'data_f{problem_id}'
                    func_dir.mkdir(exist_ok=True)
                    
                    # Use pop_size as dimension
                    dimension = pop_size
                    
                    # Create unique .dat filename for this algorithm/dimension combo
                    dat_filename = f'IOHprofiler_f{problem_id}_DIM{dimension}_{algorithm_name}.dat'
                    dat_path = func_dir / dat_filename
                    
                    best_f_vals_list = mech_data['best_f_vals']
                    
                    # Write .dat file with proper legacy format
                    with open(dat_path, 'w') as f:
                        for run_id, fitness_progression in enumerate(best_f_vals_list, start=1):
                            if not fitness_progression:
                                continue
                            
                            # Write separation line (mandatory format with quotes)
                            f.write('"function evaluation" "best-so-far f(x)"\n')
                            
                            # Get improvements
                            improvements = self._get_improvements(fitness_progression, maximization)
                            
                            # Write data for this run
                            for eval_count, fitness_value in improvements:
                                if np.isfinite(fitness_value):
                                    f.write(f'{eval_count} {fitness_value:+.5e}\n')
                    
                    # Create .info file for this algorithm
                    info_filename = f'IOHprofiler_f{problem_id}_{algorithm_name}.info'
                    info_path = self.output_dir / info_filename
                    
                    # Calculate summary statistics for each run
                    summary_data = []
                    for run_id, fitness_progression in enumerate(best_f_vals_list, start=1):
                        if not fitness_progression:
                            continue
                        
                        # Get best value
                        finite_vals = [v for v in fitness_progression if np.isfinite(v)]
                        if not finite_vals:
                            continue
                            
                        if maximization:
                            best_value = max(finite_vals)
                        else:
                            best_value = min(finite_vals)
                        
                        total_evals = len(fitness_progression)
                        # Format: instance:evals|best_value
                        summary_data.append(f"1:{total_evals}|{best_value:.5e}")
                    
                    # Write .info file with proper format
                    with open(info_path, 'w') as f:
                        # Line 1: metadata
                        f.write(f"suite = 'SubmodularOpt', funcId = {problem_id}, DIM = {dimension}, algId = '{algorithm_name}'\n")
                        # Line 2: comment (empty)
                        f.write("%\n")
                        # Line 3: data file path and run summaries
                        f.write(f"{func_dir.name}/{dat_filename}, " + ", ".join(summary_data) + "\n")
                    
                    print(f"  ✓ Created: {info_filename} ({len(summary_data)} runs)")
        
        print(f"\n✓ IOHprofiler format conversion complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"\nTo use in IOHanalyzer:")
        print(f"  1. Zip the '{self.output_dir}' folder: zip -r ioh_data.zip {self.output_dir}/")
        print(f"  2. Upload ioh_data.zip to IOHanalyzer")
        print(f"  3. Do NOT check 'use custom csv format'")
        print(f"  4. IOHanalyzer will automatically detect the .info files")



def main():
    """Main conversion script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert single_objective.py results to IOHanalyzer format'
    )
    parser.add_argument(
        'pickle_file',
        nargs='?',
        default='results_exercise3/all_results.pkl',
        help='Path to all_results.pkl file (default: results_exercise3/all_results.pkl)'
    )
    parser.add_argument(
        '--output-dir',
        default='ioh_data',
        help='Output directory for IOHanalyzer data (default: ioh_data)'
    )
    parser.add_argument(
        '--format',
        choices=['csv', 'json', 'iohprofiler', 'both', 'all'],
        default='iohprofiler',
        help='Output format (default: iohprofiler - most reliable)'
    )
    parser.add_argument(
        '--minimization',
        action='store_true',
        help='Set if this is a minimization problem (default: maximization)'
    )
    parser.add_argument(
        '--no-downsample',
        action='store_true',
        help='Disable downsampling (record all evaluations instead of only improvements)'
    )
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not os.path.exists(args.pickle_file):
        print(f"Error: Pickle file not found: {args.pickle_file}")
        return
    
    # Create converter
    converter = IOHAnalyzerConverter(args.pickle_file, args.output_dir)
    
    maximization = not args.minimization
    downsample = not args.no_downsample
    
    # Convert to requested format(s)
    if args.format in ['csv', 'both', 'all']:
        df = converter.to_csv('benchmark_results.csv', maximization=maximization, 
                            downsample=downsample)
        
        # Also create combined CSV with diversity
        converter.create_combined_csv('combined_results.csv', maximization=maximization,
                                     downsample=downsample)
    
    if args.format in ['json', 'both', 'all']:
        converter.to_json(maximization=maximization)
    
    if args.format in ['iohprofiler', 'all']:
        converter.to_iohprofiler_format(maximization=maximization)
    
    print("\n" + "="*70)
    print("Conversion complete!")
    print("="*70)


if __name__ == '__main__':
    main()