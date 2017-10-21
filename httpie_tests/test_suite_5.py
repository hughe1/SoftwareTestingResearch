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


def test_headers(httpbin_both):
    r = http('GET', httpbin_both + '/headers', 'Foo:bar')
    assert HTTP_OK in r
    assert '"User-Agent": "HTTPie' in r, r
    assert '"Foo": "bar"' in r


def test_3xx_check_status_exits_3_and_stderr_when_stdout_redirected(
        httpbin):
    env = TestEnvironment(stdout_isatty=False)
    r = http('--check-status', '--headers',
             'GET', httpbin.url + '/status/301',
             env=env, error_exit_ok=True)
    assert '301 MOVED PERMANENTLY' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_3XX
    assert '301 moved permanently' in r.stderr.lower()


def test_POST_JSON_data(httpbin_both):
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


def test_follow_all_output_options_used_for_redirects(httpbin):
    r = http('--check-status',
             '--follow',
             '--all',
             '--print=H',
             httpbin.url + '/redirect/2')
    assert r.count('GET /') == 3
    assert HTTP_OK not in r


@mock.patch('httpie.input.AuthCredentials._getpass',
            new=lambda self, prompt: 'password')
def test_password_prompt(httpbin):
    r = http('--auth', 'user',
             'GET', httpbin.url + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_unicode_url_query_arg_item(httpbin):
    r = http(httpbin.url + '/get', u'test==%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}, r


def test_unicode_json_item_verbose(httpbin):
    r = http('--verbose', '--json',
             'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


# @pytest.mark.skipif(not has_docutils(), reason='docutils not installed')
# @pytest.mark.parametrize('filename', filenames)
def test_GET(httpbin_both):
    r = http('GET', httpbin_both + '/get')
    assert HTTP_OK in r


def test_max_redirects(httpbin):
    r = http('--max-redirects=1', '--follow', httpbin.url + '/redirect/3',
             error_exit_ok=True)
    assert r.exit_status == ExitStatus.ERROR_TOO_MANY_REDIRECTS

def test_POST_stdin(httpbin_both):
    with open(FILE_PATH) as f:
        env = TestEnvironment(stdin=f, stdin_isatty=False)
        r = http('--form', 'POST', httpbin_both + '/post', env=env)
    assert HTTP_OK in r
    assert FILE_CONTENT in r


def test_debug():
    r = http('--debug')
    assert r.exit_status == httpie.ExitStatus.OK
    assert 'HTTPie %s' % httpie.__version__ in r.stderr


def test_basic_auth(httpbin_both):
    r = http('--auth=user:password',
             'GET', httpbin_both + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


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

    def test_POST_no_data_no_auto_headers(self, httpbin):
        # JSON headers shouldn't be automatically set for POST with no data.
        r = http('POST', httpbin.url + '/post')
        assert HTTP_OK in r
        assert '"Accept": "*/*"' in r
        assert '"Content-Type": "application/json' not in r

    def test_POST_with_data_auto_JSON_headers(self, httpbin):
        r = http('POST', httpbin.url + '/post', 'a=b')
        assert HTTP_OK in r
        assert r.json['headers']['Accept'] == JSON_ACCEPT
        assert r.json['headers']['Content-Type'] == 'application/json'



class TestVerboseFlag:
    def test_verbose(self, httpbin):
        r = http('--verbose',
                 'GET', httpbin.url + '/get', 'test-header:__test__')
        assert HTTP_OK in r
        assert r.count('__test__') == 2

    def test_verbose_form(self, httpbin):
        # https://github.com/jakubroztocil/httpie/issues/53
        r = http('--verbose', '--form', 'POST', httpbin.url + '/post',
                 'A=B', 'C=D')
        assert HTTP_OK in r
        assert 'A=B&C=D' in r

    def test_verbose_json(self, httpbin):
        r = http('--verbose',
                 'POST', httpbin.url + '/post', 'foo=bar', 'baz=bar')
        assert HTTP_OK in r
        assert '"baz": "bar"' in r

    def test_verbose_implies_all(self, httpbin):
        r = http('--verbose', '--follow', httpbin + '/redirect/1')
        assert 'GET /redirect/1 HTTP/1.1' in r
        assert 'HTTP/1.1 302 FOUND' in r
        assert 'GET /get HTTP/1.1' in r
        assert HTTP_OK in r


# def test_rst_file_syntax(filename):
#     p = subprocess.Popen(
#         ['rst2pseudoxml.py', '--report=1', '--exit-status=1', filename],
#         stderr=subprocess.PIPE,
#         stdout=subprocess.PIPE
#     )
#     err = p.communicate()[1]
#     assert p.returncode == 0, err.decode('utf8')

@pytest.mark.parametrize('follow_flag', ['--follow', '-F'])
def test_follow_without_all_redirects_hidden(httpbin, follow_flag):
    r = http(follow_flag, httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 1
    assert HTTP_OK in r


# @pytest.mark.skipif(not has_docutils(), reason='docutils not installed')
# @pytest.mark.parametrize('filename', filenames)
def test_unicode_form_item_verbose(httpbin):
    r = http('--verbose', '--form',
             'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


def test_unicode_form_item_verbose(httpbin):
    r = http('--verbose', '--form',
             'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


# @pytest.mark.skipif(not has_docutils(), reason='docutils not installed')
# @pytest.mark.parametrize('filename', filenames)
class TestPrettyOptions:
    """Test the --pretty flag handling."""

    def test_pretty_enabled_by_default(self, httpbin):
        env = TestEnvironment(colors=256)
        r = http('GET', httpbin.url + '/get', env=env)
        assert COLOR in r

    def test_pretty_enabled_by_default_unless_stdout_redirected(self, httpbin):
        r = http('GET', httpbin.url + '/get')
        assert COLOR not in r

    def test_force_pretty(self, httpbin):
        env = TestEnvironment(stdout_isatty=False, colors=256)
        r = http('--pretty=all', 'GET', httpbin.url + '/get', env=env, )
        assert COLOR in r

    def test_force_ugly(self, httpbin):
        r = http('--pretty=none', 'GET', httpbin.url + '/get')
        assert COLOR not in r

    def test_subtype_based_pygments_lexer_match(self, httpbin):
        """Test that media subtype is used if type/subtype doesn't
        match any lexer.

        """
        env = TestEnvironment(colors=256)
        r = http('--print=B', '--pretty=all', httpbin.url + '/post',
                 'Content-Type:text/foo+json', 'a=b', env=env)
        assert COLOR in r

    def test_colors_option(self, httpbin):
        env = TestEnvironment(colors=256)
        r = http('--print=B', '--pretty=colors',
                 'GET', httpbin.url + '/get', 'a=b',
                 env=env)
        # Tests that the JSON data isn't formatted.
        assert not r.strip().count('\n')
        assert COLOR in r

    def test_format_option(self, httpbin):
        env = TestEnvironment(colors=256)
        r = http('--print=B', '--pretty=format',
                 'GET', httpbin.url + '/get', 'a=b',
                 env=env)
        # Tests that the JSON data is formatted.
        assert r.strip().count('\n') == 2
        assert COLOR not in r

