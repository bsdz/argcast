import nox
from dependency_groups import resolve


def get_optional_dependencies(*groups):
    pyproject = nox.project.load_toml("pyproject.toml")
    dep_groups = pyproject["project"]["optional-dependencies"]
    return resolve(dep_groups, *groups)


@nox.session
def lint(session):
    session.install(*get_optional_dependencies("dev"))

    session.run("black", ".", "--check")
    session.run("isort", ".", "--check-only", "--profile", "black")
    session.run("docformatter", ".", "--recursive", "--check", "--diff", "--black")


@nox.session(python=["3.12", "3.13"])
def test(session):
    session.install(*get_optional_dependencies("test", "dev"))
    session.run("python", "-m", "unittest")
    session.run("coverage", "run", "-m", "unittest")
    session.run("coverage", "report", "-m")
