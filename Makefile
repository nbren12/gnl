all:
	python setup.py build_ext --inplace

clean:
	find . -name '*.so' -exec rm -i {} +
