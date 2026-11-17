cd "C:\Users\alina\OneDrive\Desktop\fermi project"
python src\download_lcr.py sources.txt --out data\raw --limit 3
python src\clean_lcr.py data\raw --inspect