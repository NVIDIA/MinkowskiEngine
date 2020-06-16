export CXX=g++-7;

rm -rf build; python setup.py install --test=coordinate
python -m unittest coordinate_test

rm -rf build; python setup.py install --test=coordinate_map_key
python -m unittest coordinate_map_key_test

rm -rf build; python setup.py install --test=coordinate_map_cpu
python -m unittest coordinate_map_cpu_test

rm -rf build; python setup.py install --test=coordinate_map_gpu
python -m unittest coordinate_map_gpu_test

rm -rf build; python setup.py install --test=region_cpu
python -m unittest kernel_region_cpu_test
