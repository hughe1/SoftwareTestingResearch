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


class TestRequestBodyFromFilePath:
    """
    `http URL @file'

    """

    def test_request_body_from_file_by_path(self, httpbin):
        r = http('--verbose',
                 'POST', httpbin.url + '/post', '@' + FILE_PATH_ARG)
        assert HTTP_OK in r
        assert FILE_CONTENT in r, r
        assert '"Content-Type": "text/plain"' in r

    def test_request_body_from_file_by_path_with_explicit_content_type(
            self, httpbin):
        r = http('--verbose',
                 'POST', httpbin.url + '/post', '@' + FILE_PATH_ARG,
                 'Content-Type:text/plain; charset=utf8')
        assert HTTP_OK in r
        assert FILE_CONTENT in r
        assert 'Content-Type: text/plain; charset=utf8' in r

    def test_request_body_from_file_by_path_no_field_name_allowed(
            self, httpbin):
        env = TestEnvironment(stdin_isatty=True)
        r = http('POST', httpbin.url + '/post', 'field-name@' + FILE_PATH_ARG,
                 env=env, error_exit_ok=True)
        assert 'perhaps you meant --form?' in r.stderr

    def test_request_body_from_file_by_path_no_data_items_allowed(
            self, httpbin):
        env = TestEnvironment(stdin_isatty=False)
        r = http('POST', httpbin.url + '/post', '@' + FILE_PATH_ARG, 'foo=bar',
                 env=env, error_exit_ok=True)
        assert 'cannot be mixed' in r.stderr

def test_follow_redirect_output_options(httpbin):
    r = http('--check-status',
             '--follow',
             '--all',
             '--print=h',
             '--history-print=H',
             httpbin.url + '/redirect/2')
    assert r.count('GET /') == 2
    assert 'HTTP/1.1 302 FOUND' not in r
    assert HTTP_OK in r


def test_5xx_check_status_exits_5(httpbin):
    r = http('--check-status', 'GET', httpbin.url + '/status/500',
             error_exit_ok=True)
    assert '500 INTERNAL SERVER ERROR' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_5XX

def test_unicode_form_item(httpbin):
    r = http('--form', 'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['form'] == {'test': UNICODE}


def test_unicode_headers_verbose(httpbin):
    # httpbin doesn't interpret utf8 headers
    r = http('--verbose', httpbin.url + '/headers', u'Test:%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


def test_POST_JSON_data(httpbin_both):
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


def test_ok_response_exits_0(httpbin):
    r = http('GET', httpbin.url + '/get')
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.OK





def test_3xx_check_status_exits_3_and_stderr_when_stdout_redirected(
        httpbin):
    env = TestEnvironment(stdout_isatty=False)
    r = http('--check-status', '--headers',
             'GET', httpbin.url + '/status/301',
             env=env, error_exit_ok=True)
    assert '301 MOVED PERMANENTLY' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_3XX
    assert '301 moved permanently' in r.stderr.lower()


def test_help():
    r = http('--help', error_exit_ok=True)
    assert r.exit_status == httpie.ExitStatus.OK
    assert 'https://github.com/jakubroztocil/httpie/issues' in r


@mock.patch('httpie.input.AuthCredentials._getpass',
            new=lambda self, prompt: 'password')
def test_password_prompt(httpbin):
    r = http('--auth', 'user',
             'GET', httpbin.url + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_error_response_exits_0_without_check_status(httpbin):
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERRO' in r
    assert r.exit_status == ExitStatus.OK
    assert not r.stderr


def test_default_options(httpbin):
    env = TestEnvironment()
    env.config['default_options'] = ['--form']
    env.config.save()
    r = http(httpbin.url + '/post', 'foo=bar', env=env)
    assert r.json['form'] == {"foo": "bar"}


def test_POST_stdin(httpbin_both):
    with open(FILE_PATH) as f:
        env = TestEnvironment(stdin=f, stdin_isatty=False)
        r = http('--form', 'POST', httpbin_both + '/post', env=env)
    assert HTTP_OK in r
    assert FILE_CONTENT in r


def test_follow_all_redirects_shown(httpbin):
    r = http('--follow', '--all', httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 3
    assert r.count('HTTP/1.1 302 FOUND', 2)
    assert HTTP_OK in r


# class TestSession(SessionTestBase):
#     """Stand-alone session tests."""
# 
#     def test_session_ignored_header_prefixes(self, httpbin):
#         self.start_session(httpbin)
#         r1 = http('--session=test', 'GET', httpbin.url + '/get',
#                   'Content-Type: text/plain',
#                   'If-Unmodified-Since: Sat, 29 Oct 1994 19:43:31 GMT',
#                   env=self.env())
#         assert HTTP_OK in r1
#         r2 = http('--session=test', 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r2
#         assert 'Content-Type' not in r2.json['headers']
#         assert 'If-Unmodified-Since' not in r2.json['headers']
# 
#     def test_session_by_path(self, httpbin):
#         self.start_session(httpbin)
#         session_path = os.path.join(self.config_dir, 'session-by-path.json')
#         r1 = http('--session=' + session_path, 'GET', httpbin.url + '/get',
#                   'Foo:Bar', env=self.env())
#         assert HTTP_OK in r1
# 
#         r2 = http('--session=' + session_path, 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r2
#         assert r2.json['headers']['Foo'] == 'Bar'
# 
#     @pytest.mark.skipif(
#         sys.version_info >= (3,),
#         reason="This test fails intermittently on Python 3 - "
#                "see https://github.com/jakubroztocil/httpie/issues/282")
#     def test_session_unicode(self, httpbin):
#         self.start_session(httpbin)
# 
#         r1 = http('--session=test', u'--auth=test:' + UNICODE,
#                   'GET', httpbin.url + '/get', u'Test:%s' % UNICODE,
#                   env=self.env())
#         assert HTTP_OK in r1
# 
#         r2 = http('--session=test', '--verbose', 'GET',
#                   httpbin.url + '/get', env=self.env())
#         assert HTTP_OK in r2
# 
#         # FIXME: Authorization *sometimes* is not present on Python3
#         assert (r2.json['headers']['Authorization'] ==
#                 HTTPBasicAuth.make_header(u'test', UNICODE))
#         # httpbin doesn't interpret utf8 headers
#         assert UNICODE in r2
# 
#     def test_session_default_header_value_overwritten(self, httpbin):
#         self.start_session(httpbin)
#         # https://github.com/jakubroztocil/httpie/issues/180
#         r1 = http('--session=test',
#                   httpbin.url + '/headers', 'User-Agent:custom',
#                   env=self.env())
#         assert HTTP_OK in r1
#         assert r1.json['headers']['User-Agent'] == 'custom'
# 
#         r2 = http('--session=test', httpbin.url + '/headers', env=self.env())
#         assert HTTP_OK in r2
#         assert r2.json['headers']['User-Agent'] == 'custom'
# 
#     def test_download_in_session(self, httpbin):
#         # https://github.com/jakubroztocil/httpie/issues/412
#         self.start_session(httpbin)
#         cwd = os.getcwd()
#         os.chdir(gettempdir())
#         try:
#             http('--session=test', '--download',
#                  httpbin.url + '/get', env=self.env())
#         finally:
#             os.chdir(cwd)

def test_unicode_url_query_arg_item_verbose(httpbin):
    r = http('--verbose', httpbin.url + '/get', u'test==%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


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


def test_DELETE(httpbin_both):
    r = http('DELETE', httpbin_both + '/delete')
    assert HTTP_OK in r


@pytest.mark.parametrize('stdout_isatty', [True, False])
def test_output_option(httpbin, stdout_isatty):
    output_filename = os.path.join(gettempdir(), test_output_option.__name__)
    url = httpbin + '/robots.txt'

    r = http('--output', output_filename, url,
             env=TestEnvironment(stdout_isatty=stdout_isatty))
    assert r == ''

    expected_body = urlopen(url).read().decode()
    with open(output_filename, 'r') as f:
        actual_body = f.read()

    assert actual_body == expected_body


def test_redirected_stream(httpbin):
    """Test that --stream works with non-prettified
    redirected terminal output."""
    with open(BIN_FILE_PATH, 'rb') as f:
        env = TestEnvironment(stdout_isatty=False,
                              stdin_isatty=False,
                              stdin=f)
        r = http('--pretty=none', '--stream', '--verbose', 'GET',
                 httpbin.url + '/get', env=env)
    assert BIN_FILE_CONTENT in r
