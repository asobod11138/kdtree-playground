ifndef SOURCE_FILES
	export SOURCE_FILES:=kdtree_lib
endif

ifndef TEST_FILES
	export TEST_FILES:=tests
endif

.PHONY: lint test format format-apply publish

format:
	poetry run ruff check ${SOURCE_FILES} --exit-zero
	poetry run ruff check ${TEST_FILES} --ignore ANN
	poetry run ruff format --diff ${SOURCE_FILES}
	poetry run ruff format --diff ${TEST_FILES}

format-apply:
	poetry run ruff check ${SOURCE_FILES} ${TEST_FILES} --fix --ignore ANN
	poetry run ruff format ${SOURCE_FILES} ${TEST_FILES}

lint:
	poetry run ruff ${SOURCE_FILES}
	# テストコードはチェックを緩和する
	# pygrep-hooks, flake8-datetimez, line-too-long, flake8-annotations, unused-noqa
	poetry run ruff check ${TEST_FILES} --ignore PGH,DTZ,E501,ANN,RUF100

test:
	poetry run pytest

mypy:
	poetry run mypy ${SOURCE_FILES}