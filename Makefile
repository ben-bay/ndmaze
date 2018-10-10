
clean-pyc:
	find . -name '*.pyc' -exec rm '{}' +
	find . -name '*.pyo' -exec rm '{}' +
	#name '*~' -exec rm '{}' 

install:
	( \
		python3 -m virtualenv venv; \
        source venv/bin/activate; \
        pip3 install -r requirements.txt; \
    )

test: clean-pyc
	py.test

lint:
	( \
		source venv/bin/activate; \
		flake8 src; \
	)

update:
	git pull; install
