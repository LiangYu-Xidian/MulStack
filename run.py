#! /usr/bin/env python

from webserver import app

app.run(debug=True, host='localhost', port=5000)
