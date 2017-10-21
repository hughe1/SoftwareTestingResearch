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


@mock.patch('httpie.input.AuthCredentials._getpass',
            new=lambda self, prompt: 'UNEXPECTED_PROMPT_RESPONSE')
def test_auth_plugin_prompt_password_false(httpbin):

    class Plugin(AuthPlugin):
        auth_type = 'test-prompt-false'
        prompt_password = False

        def get_auth(self, username=None, password=None):
            assert self.raw_auth == USERNAME
            assert username == USERNAME
            assert password is None
            return basic_auth()

    plugin_manager.register(Plugin)

    try:
        r = http(
            httpbin + BASIC_AUTH_URL,
            '--auth-type',
            Plugin.auth_type,
            '--auth',
            USERNAME,
        )
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)

def test_POST_stdin(httpbin_both):
    with open(FILE_PATH) as f:
        env = TestEnvironment(stdin=f, stdin_isatty=False)
        r = http('--form', 'POST', httpbin_both + '/post', env=env)
    assert HTTP_OK in r
    assert FILE_CONTENT in r


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

def test_POST_form_multiple_values(httpbin_both):
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar', 'foo=baz')
    assert HTTP_OK in r
    assert r.json['form'] == {'foo': ['bar', 'baz']}


class TestImplicitHTTPMethod:
    def test_implicit_GET(self, httpbin):
        r = http(httpbin.url + '/get')
        assert HTTP_OK in r

    def test_implicit_GET_with_headers(self, httpbin):
        r = http(httpbin.url + '/headers', 'Foo:bar')
        assert HTTP_OK in r
        assert r.json['headers']['Foo'] == 'bar'

    def test_implicit_POST_json(self, httpbin):
        r = http(httpbin.url + '/post', 'hello=world')
        assert HTTP_OK in r
        assert r.json['json'] == {'hello': 'world'}

    def test_implicit_POST_form(self, httpbin):
        r = http('--form', httpbin.url + '/post', 'foo=bar')
        assert HTTP_OK in r
        assert r.json['form'] == {'foo': 'bar'}

    def test_implicit_POST_stdin(self, httpbin):
        with open(FILE_PATH) as f:
            env = TestEnvironment(stdin_isatty=False, stdin=f)
            r = http('--form', httpbin.url + '/post', env=env)
        assert HTTP_OK in r


