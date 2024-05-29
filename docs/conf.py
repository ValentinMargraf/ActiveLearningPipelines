import automl_sphinx_theme

from ALP import author, copyright, name, version

options = {
    "copyright": copyright,
    "author": author,
    "version": version,
    "name": name,
    "html_theme_options": {
        "github_url": "https://github.com/ValentinMargraf/ActiveLearningPipelines",
        "twitter_url": "https://twitter.com/automl_org?lang=de",
    },
}

automl_sphinx_theme.set_options(globals(), options)
