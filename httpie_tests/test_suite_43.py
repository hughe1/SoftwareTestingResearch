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


@mock.patch('httpie.core.get_response')
def test_error(get_response):
    def error(msg, *args, **kwargs):
        global error_msg
        error_msg = msg % args

    exc = ConnectionError('Connection aborted')
    exc.request = Request(method='GET', url='http://www.google.com')
    get_response.side_effect = exc
    ret = main(['--ignore-stdin', 'www.google.com'], custom_log_error=error)
    assert ret == ExitStatus.ERROR
    assert error_msg == (
        'ConnectionError: '
        'Connection aborted while doing GET request to URL: '
        'http://www.google.com')


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


def test_help():
    r = http('--help', error_exit_ok=True)
    assert r.exit_status == httpie.ExitStatus.OK
    assert 'https://github.com/jakubroztocil/httpie/issues' in r


def test_credentials_in_url_auth_flag_has_priority(httpbin_both):
    """When credentials are passed in URL and via -a at the same time,
     then the ones from -a are used."""
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:wrong')
    r = http('--auth=user:password', 'GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_unicode_raw_json_item_verbose(httpbin):
    r = http('--json', 'POST', httpbin.url + '/post',
             u'test:={ "%s" : [ "%s" ] }' % (UNICODE, UNICODE))
    assert HTTP_OK in r
    assert r.json['json'] == {'test': {UNICODE: [UNICODE]}}


def test_error_response_exits_0_without_check_status(httpbin):
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERRO' in r
    assert r.exit_status == ExitStatus.OK
    assert not r.stderr


def test_migrate_implicit_content_type():
    config = TestEnvironment().config

    config['implicit_content_type'] = 'json'
    config.save()
    config.load()
    assert 'implicit_content_type' not in config
    assert not config['default_options']

    config['implicit_content_type'] = 'form'
    config.save()
    config.load()
    assert 'implicit_content_type' not in config
    assert config['default_options'] == ['--form']

@pytest.mark.parametrize('url', [
    'username@example.org',
    'username:@example.org',
])
def test_only_username_in_url(url):
    """
    https://github.com/jakubroztocil/httpie/issues/242

    """
    args = httpie.cli.parser.parse_args(args=[url], env=TestEnvironment())
    assert args.auth
    assert args.auth.username == 'username'
    assert args.auth.password == ''


def test_3xx_check_status_redirects_allowed_exits_0(httpbin):
    r = http('--check-status', '--follow',
             'GET', httpbin.url + '/status/301',
             error_exit_ok=True)
    # The redirect will be followed so 200 is expected.
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.OK


def basic_auth(header=BASIC_AUTH_HEADER_VALUE):

    def inner(r):
        r.headers['Authorization'] = header
        return r

    return inner

