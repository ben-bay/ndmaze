
clean-pyc:
	find . -name '*.pyc' -exec rm --force {} + find . -name '*.pyo' -exec rm --force {} + name '*~' -exec rm --force  {} 

install:
	( \
		python3 -m virtualenv venv; \
        source venv/bin/activate; \
        pip3 install -r requirements.txt; \
    )

test: clean-pyc
	py.test
