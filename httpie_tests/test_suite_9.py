"""HTTP authentication-related tests."""
import mock
import pytest

from utils import http, add_auth, HTTP_OK, TestEnvironment
import httpie.input
import httpie.cli

import os
from tempfile import gettempdir

import pytest

from utils import TestEnvironment, http, HTTP_OK, COLOR, CRLF
from httpie import ExitStatus
from httpie.compat import urlopen
from httpie.output.formatters.colors import get_lexer

"""High-level tests."""
import pytest

from httpie.input import ParseError
from utils import TestEnvironment, http, HTTP_OK
from fixtures import FILE_PATH, FILE_CONTENT

import httpie
from httpie.compat import is_py26

import os

import pytest

from httpie.input import ParseError
from utils import TestEnvironment, http, HTTP_OK
from fixtures import FILE_PATH_ARG, FILE_PATH, FILE_CONTENT

import os
import tempfile

import pytest
from httpie.context import Environment

from utils import TestEnvironment, http
from httpie.compat import is_windows

"""
Tests for the provided defaults regarding HTTP method, and --json vs. --form.

"""
from httpie.client import JSON_ACCEPT
from utils import TestEnvironment, http, HTTP_OK
from fixtures import FILE_PATH

# coding=utf-8
import os
import shutil
import sys
from tempfile import gettempdir

import pytest

from httpie.plugins.builtin import HTTPBasicAuth
from utils import TestEnvironment, mk_config_dir, http, HTTP_OK
from fixtures import UNICODE

from utils import TestEnvironment, http

from mock import mock

from httpie.input import SEP_CREDENTIALS
from httpie.plugins import AuthPlugin, plugin_manager
from utils import http, HTTP_OK

# TODO: run all these tests in session mode as well

USERNAME = 'user'
PASSWORD = 'password'
# Basic auth encoded `USERNAME` and `PASSWORD`
# noinspection SpellCheckingInspection
BASIC_AUTH_HEADER_VALUE = 'Basic dXNlcjpwYXNzd29yZA=='
BASIC_AUTH_URL = '/basic-auth/{0}/{1}'.format(USERNAME, PASSWORD)
AUTH_OK = {'authenticated': True, 'user': USERNAME}

"""Tests for dealing with binary request and response data."""
from fixtures import BIN_FILE_PATH, BIN_FILE_CONTENT, BIN_FILE_PATH_ARG
from httpie.compat import urlopen
from httpie.output.streams import BINARY_SUPPRESSED_NOTICE
from utils import TestEnvironment, http

"""High-level tests."""
import pytest

from httpie import ExitStatus
from utils import http, HTTP_OK

import mock
from pytest import raises
from requests import Request, Timeout
from requests.exceptions import ConnectionError

from httpie import ExitStatus
from httpie.core import main

error_msg = None

import os
import time

import pytest
import mock
from requests.structures import CaseInsensitiveDict

from httpie.compat import urlopen
from httpie.downloads import (
    parse_content_range, filename_from_content_disposition, filename_from_url,
    get_unique_filename, ContentRangeError, Downloader,
)
from utils import http, TestEnvironment

# coding=utf-8
"""
Various unicode handling related tests.

"""
from utils import http, HTTP_OK
from fixtures import UNICODE

import os
import fnmatch
import subprocess

import pytest

from utils import TESTS_ROOT

"""Miscellaneous regression tests"""
import pytest

from utils import http, HTTP_OK
from httpie.compat import is_windows

"""CLI argument parsing related tests."""
import json
# noinspection PyCompatibility
import argparse

import pytest
from requests.exceptions import InvalidSchema

from httpie import input
from httpie.input import KeyValue, KeyValueArgType, DataDict
from httpie import ExitStatus
from httpie.cli import parser
from utils import TestEnvironment, http, HTTP_OK
from fixtures import (
    FILE_PATH_ARG, JSON_FILE_PATH_ARG,
    JSON_FILE_CONTENT, FILE_CONTENT, FILE_PATH
)

import pytest

from httpie.compat import is_windows
from httpie.output.streams import BINARY_SUPPRESSED_NOTICE
from utils import http, TestEnvironment
from fixtures import BIN_FILE_CONTENT, BIN_FILE_PATH


# GET because httpbin 500s with binary POST body.

import mock

from httpie import ExitStatus
from utils import TestEnvironment, http, HTTP_OK


def test_4xx_check_status_exits_4(httpbin):
    r = http('--check-status', 'GET', httpbin.url + '/status/401',
             error_exit_ok=True)
    assert '401 UNAUTHORIZED' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_4XX
    # Also stderr should be empty since stdout isn't redirected.
    assert not r.stderr


def test_unicode_url(httpbin):
    r = http(httpbin.url + u'/get?test=' + UNICODE)
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}

# def test_unicode_url_verbose(self):
#     r = http(httpbin.url + '--verbose', u'/get?test=' + UNICODE)
#     assert HTTP_OK in r


class TestAutoContentTypeAndAcceptHeaders:
    """
    Test that Accept and Content-Type correctly defaults to JSON,
    but can still be overridden. The same with Content-Type when --form
    -f is used.

    """

    def test_GET_no_data_no_auto_headers(self, httpbin):
        # https://github.com/jakubroztocil/httpie/issues/62
        r = http('GET', httpbin.url + '/headers')
        assert HTTP_OK in r
        assert r.json['headers']['Accept'] == '*/*'
        assert 'Content-Type' not in r.json['headers']

