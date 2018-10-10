
.PHONY : clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm '{}' +
	find . -name '*.pyo' -exec rm '{}' +
	#name '*~' -exec rm '{}' 

.PHONY : install
install:
	( \
		python3 -m virtualenv venv; \
        source venv/bin/activate; \
        pip3 install -r requirements.txt; \
    )

.PHONY : test
test: clean-pyc
	py.test

.PHONY : lint
lint:
	( \
		source venv/bin/activate; \
		flake8 src; \
	)

.PHONY : pull
pull:
	git pull

.PHONY: update
update:	pull install
