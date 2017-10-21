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


def test_unicode_json_item(httpbin):
    r = http('--json', 'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['json'] == {'test': UNICODE}


def basic_auth(header=BASIC_AUTH_HEADER_VALUE):

    def inner(r):
        r.headers['Authorization'] = header
        return r

    return inner


def test_credentials_in_url(httpbin_both):
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:password')
    r = http('GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


class TestArgumentParser:

    def setup_method(self, method):
        self.parser = input.HTTPieArgumentParser()

    def test_guess_when_method_set_and_valid(self):
        self.parser.args = argparse.Namespace()
        self.parser.args.method = 'GET'
        self.parser.args.url = 'http://example.com/'
        self.parser.args.items = []
        self.parser.args.ignore_stdin = False

        self.parser.env = TestEnvironment()

        self.parser._guess_method()

        assert self.parser.args.method == 'GET'
        assert self.parser.args.url == 'http://example.com/'
        assert self.parser.args.items == []

    def test_guess_when_method_not_set(self):
        self.parser.args = argparse.Namespace()
        self.parser.args.method = None
        self.parser.args.url = 'http://example.com/'
        self.parser.args.items = []
        self.parser.args.ignore_stdin = False
        self.parser.env = TestEnvironment()

        self.parser._guess_method()

        assert self.parser.args.method == 'GET'
        assert self.parser.args.url == 'http://example.com/'
        assert self.parser.args.items == []

    def test_guess_when_method_set_but_invalid_and_data_field(self):
        self.parser.args = argparse.Namespace()
        self.parser.args.method = 'http://example.com/'
        self.parser.args.url = 'data=field'
        self.parser.args.items = []
        self.parser.args.ignore_stdin = False
        self.parser.env = TestEnvironment()
        self.parser._guess_method()

        assert self.parser.args.method == 'POST'
        assert self.parser.args.url == 'http://example.com/'
        assert self.parser.args.items == [
            KeyValue(key='data',
                     value='field',
                     sep='=',
                     orig='data=field')
        ]

    def test_guess_when_method_set_but_invalid_and_header_field(self):
        self.parser.args = argparse.Namespace()
        self.parser.args.method = 'http://example.com/'
        self.parser.args.url = 'test:header'
        self.parser.args.items = []
        self.parser.args.ignore_stdin = False

        self.parser.env = TestEnvironment()

        self.parser._guess_method()

        assert self.parser.args.method == 'GET'
        assert self.parser.args.url == 'http://example.com/'
        assert self.parser.args.items, [
            KeyValue(key='test',
                     value='header',
                     sep=':',
                     orig='test:header')
        ]

    def test_guess_when_method_set_but_invalid_and_item_exists(self):
        self.parser.args = argparse.Namespace()
        self.parser.args.method = 'http://example.com/'
        self.parser.args.url = 'new_item=a'
        self.parser.args.items = [
            KeyValue(
                key='old_item', value='b', sep='=', orig='old_item=b')
        ]
        self.parser.args.ignore_stdin = False

        self.parser.env = TestEnvironment()

        self.parser._guess_method()

        assert self.parser.args.items, [
            KeyValue(key='new_item', value='a', sep='=', orig='new_item=a'),
            KeyValue(
                key='old_item', value='b', sep='=', orig='old_item=b'),
        ]


class TestSchemes:

    def test_invalid_custom_scheme(self):
        # InvalidSchema is expected because HTTPie
        # shouldn't touch a formally valid scheme.
        with pytest.raises(InvalidSchema):
            http('foo+bar-BAZ.123://bah')

    def test_invalid_scheme_via_via_default_scheme(self):
        # InvalidSchema is expected because HTTPie
        # shouldn't touch a formally valid scheme.
        with pytest.raises(InvalidSchema):
            http('bah', '--default=scheme=foo+bar-BAZ.123')

    def test_default_scheme(self, httpbin_secure):
        url = '{0}:{1}'.format(httpbin_secure.host, httpbin_secure.port)
        assert HTTP_OK in http(url, '--default-scheme=https')

def test_Host_header_overwrite(httpbin):
    """
    https://github.com/jakubroztocil/httpie/issues/235

    """
    host = 'httpbin.org'
    url = httpbin.url + '/get'
    r = http('--print=hH', url, 'host:{0}'.format(host))
    assert HTTP_OK in r
    assert r.lower().count('host:') == 1
    assert 'host: {0}'.format(host) in r

