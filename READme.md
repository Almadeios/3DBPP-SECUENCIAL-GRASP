# Procedimiento para usar el programa 
 * Crear un enviroment Python 3.11.6
 * .\.venv\Scripts\activate
 * pip install -r requirements.txt

# Ejecución principal (empaque secuencial mejorado)
Usa `main.py`, que soporta metaheurísticas (`--metaheuristic grasp/tabu/sa/none`), buffer K y distintos datasets (general, kitchen, blockout).

Ejemplo básico (grasp, k=3, step=0.02, dataset blockout):
```
python main.py --dataset blockout --buffer-size 3 --step 0.02 --sequence-index 3 --restrict-rotations --zhao-order --metaheuristic grasp --grasp-iterations 10 --rcl-size 8 --max-passes 2 --random-seed 42
```
Salida: `resultados/<dataset>/solucion_<meta>_k<k>_s####.json` y metadata en `resultados/<dataset>/meta/`.

Estos resultados pueden ser observados 3D mediante:
```
python vista_secuencial.py --json solucion_grasp_k3_s0020.json
```

# Scripts de experimentos
- `run_experiments_kitchen.py`, `run_experiments_blockout.py`: barren buffers [1,3,5,10] y steps [0.01,0.02,0.03] con configuración fija por dataset. Ejecuta desde la raíz:
```
python run_experiments_kitchen.py
python run_experiments_blockout.py
```
# Los datasets estan disponibles en este link
```
https://drive.google.com/drive/folders/1TibQqFfzugui1gBI_wIcW6H6CzF_cIwj
```



AÑADIR PARALELISMO

ADEMAS EXTENDER EL GRASP ITERATIONS PERO DETENERLO SI X VECES LOS RESULTADOS NO SON SIGNIFICATIVAMENTE MEJORES.
