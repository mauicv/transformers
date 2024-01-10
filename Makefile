run_mof_profiler: 
	python src/benchmarks/mof.py

graph_profile:
	gprof2dot -f pstats src/benchmarks/results/mof.prof | dot -Tpng -o profile-img.png

run_profiles:
	python src/benchmarks/profiles.py
