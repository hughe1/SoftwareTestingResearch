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
            new=lambda self, prompt: 'password')
def test_password_prompt(httpbin):
    r = http('--auth', 'user',
             'GET', httpbin.url + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_auth_plugin_require_auth_false(httpbin):

    class Plugin(AuthPlugin):
        auth_type = 'test-require-false'
        auth_require = False

        def get_auth(self, username=None, password=None):
            assert self.raw_auth is None
            assert username is None
            assert password is None
            return basic_auth()

    plugin_manager.register(Plugin)
    try:
        r = http(
            httpbin + BASIC_AUTH_URL,
            '--auth-type',
            Plugin.auth_type,
        )
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)


class TestDownloadUtils:
    def test_Content_Range_parsing(self):
        parse = parse_content_range

        assert parse('bytes 100-199/200', 100) == 200
        assert parse('bytes 100-199/*', 100) == 200

        # missing
        pytest.raises(ContentRangeError, parse, None, 100)

        # syntax error
        pytest.raises(ContentRangeError, parse, 'beers 100-199/*', 100)

        # unexpected range
        pytest.raises(ContentRangeError, parse, 'bytes 100-199/*', 99)

        # invalid instance-length
        pytest.raises(ContentRangeError, parse, 'bytes 100-199/199', 100)

        # invalid byte-range-resp-spec
        pytest.raises(ContentRangeError, parse, 'bytes 100-99/199', 100)

        # invalid byte-range-resp-spec
        pytest.raises(ContentRangeError, parse, 'bytes 100-100/*', 100)

    @pytest.mark.parametrize('header, expected_filename', [
        ('attachment; filename=hello-WORLD_123.txt', 'hello-WORLD_123.txt'),
        ('attachment; filename=".hello-WORLD_123.txt"', 'hello-WORLD_123.txt'),
        ('attachment; filename="white space.txt"', 'white space.txt'),
        (r'attachment; filename="\"quotes\".txt"', '"quotes".txt'),
        ('attachment; filename=/etc/hosts', 'hosts'),
        ('attachment; filename=', None)
    ])
    def test_Content_Disposition_parsing(self, header, expected_filename):
        assert filename_from_content_disposition(header) == expected_filename

    def test_filename_from_url(self):
        assert 'foo.txt' == filename_from_url(
            url='http://example.org/foo',
            content_type='text/plain'
        )
        assert 'foo.html' == filename_from_url(
            url='http://example.org/foo',
            content_type='text/html; charset=utf8'
        )
        assert 'foo' == filename_from_url(
            url='http://example.org/foo',
            content_type=None
        )
        assert 'foo' == filename_from_url(
            url='http://example.org/foo',
            content_type='x-foo/bar'
        )

    @pytest.mark.parametrize(
        'orig_name, unique_on_attempt, expected',
        [
            # Simple
            ('foo.bar', 0, 'foo.bar'),
            ('foo.bar', 1, 'foo.bar-1'),
            ('foo.bar', 10, 'foo.bar-10'),
            # Trim
            ('A' * 20, 0, 'A' * 10),
            ('A' * 20, 1, 'A' * 8 + '-1'),
            ('A' * 20, 10, 'A' * 7 + '-10'),
            # Trim before ext
            ('A' * 20 + '.txt', 0, 'A' * 6 + '.txt'),
            ('A' * 20 + '.txt', 1, 'A' * 4 + '.txt-1'),
            # Trim at the end
            ('foo.' + 'A' * 20, 0, 'foo.' + 'A' * 6),
            ('foo.' + 'A' * 20, 1, 'foo.' + 'A' * 4 + '-1'),
            ('foo.' + 'A' * 20, 10, 'foo.' + 'A' * 3 + '-10'),
        ]
    )
    @mock.patch('httpie.downloads.get_filename_max_length')
    def test_unique_filename(self, get_filename_max_length,
                             orig_name, unique_on_attempt,
                             expected):

        def attempts(unique_on_attempt=0):
            # noinspection PyUnresolvedReferences,PyUnusedLocal
            def exists(filename):
                if exists.attempt == unique_on_attempt:
                    return False
                exists.attempt += 1
                return True

            exists.attempt = 0
            return exists

        get_filename_max_length.return_value = 10

        actual = get_unique_filename(orig_name, attempts(unique_on_attempt))
        assert expected == actual


def test_unicode_url_query_arg_item(httpbin):
    r = http(httpbin.url + '/get', u'test==%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}, r


def test_credentials_in_url(httpbin_both):
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:password')
    r = http('GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


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

def test_auth_plugin_require_auth_false(httpbin):

    class Plugin(AuthPlugin):
        auth_type = 'test-require-false'
        auth_require = False

        def get_auth(self, username=None, password=None):
            assert self.raw_auth is None
            assert username is None
            assert password is None
            return basic_auth()

    plugin_manager.register(Plugin)
    try:
        r = http(
            httpbin + BASIC_AUTH_URL,
            '--auth-type',
            Plugin.auth_type,
        )
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)


# class TestSessionFlow(SessionTestBase):
#     """
#     These tests start with an existing session created in `setup_method()`.
# 
#     """
# 
#     def start_session(self, httpbin):
#         """
#         Start a full-blown session with a custom request header,
#         authorization, and response cookies.
# 
#         """
#         super(TestSessionFlow, self).start_session(httpbin)
#         r1 = http('--follow', '--session=test', '--auth=username:password',
#                   'GET', httpbin.url + '/cookies/set?hello=world',
#                   'Hello:World',
#                   env=self.env())
#         assert HTTP_OK in r1
# 
#     def test_session_created_and_reused(self, httpbin):
#         self.start_session(httpbin)
#         # Verify that the session created in setup_method() has been used.
#         r2 = http('--session=test',
#                   'GET', httpbin.url + '/get', env=self.env())
#         assert HTTP_OK in r2
#         assert r2.json['headers']['Hello'] == 'World'
#         assert r2.json['headers']['Cookie'] == 'hello=world'
#         assert 'Basic ' in r2.json['headers']['Authorization']
# 
#     def test_session_update(self, httpbin):
#         self.start_session(httpbin)
#         # Get a response to a request from the original session.
#         r2 = http('--session=test', 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r2
# 
#         # Make a request modifying the session data.
#         r3 = http('--follow', '--session=test', '--auth=username:password2',
#                   'GET', httpbin.url + '/cookies/set?hello=world2',
#                   'Hello:World2',
#                   env=self.env())
#         assert HTTP_OK in r3
# 
#         # Get a response to a request from the updated session.
#         r4 = http('--session=test', 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r4
#         assert r4.json['headers']['Hello'] == 'World2'
#         assert r4.json['headers']['Cookie'] == 'hello=world2'
#         assert (r2.json['headers']['Authorization'] !=
#                 r4.json['headers']['Authorization'])
# 
#     def test_session_read_only(self, httpbin):
#         self.start_session(httpbin)
#         # Get a response from the original session.
#         r2 = http('--session=test', 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r2
# 
#         # Make a request modifying the session data but
#         # with --session-read-only.
#         r3 = http('--follow', '--session-read-only=test',
#                   '--auth=username:password2', 'GET',
#                   httpbin.url + '/cookies/set?hello=world2', 'Hello:World2',
#                   env=self.env())
#         assert HTTP_OK in r3
# 
#         # Get a response from the updated session.
#         r4 = http('--session=test', 'GET', httpbin.url + '/get',
#                   env=self.env())
#         assert HTTP_OK in r4
# 
#         # Origin can differ on Travis.
#         del r2.json['origin'], r4.json['origin']
#         # Different for each request.
# 
#         # Should be the same as before r3.
#         assert r2.json == r4.json


@pytest.mark.skip('unimplemented')
def test_unset_host_header(httpbin_both):
    r = http('GET', httpbin_both + '/headers')
    assert 'Host' in r.json['headers']  # default Host present

    r = http('GET', httpbin_both + '/headers', 'Host:')
    assert 'Host' not in r.json['headers']   # default Host unset


class TestIgnoreStdin:

    def test_ignore_stdin(self, httpbin):
        with open(FILE_PATH) as f:
            env = TestEnvironment(stdin=f, stdin_isatty=False)
            r = http('--ignore-stdin', '--verbose', httpbin.url + '/get',
                     env=env)
        assert HTTP_OK in r
        assert 'GET /get HTTP' in r, "Don't default to POST."
        assert FILE_CONTENT not in r, "Don't send stdin data."

    def test_ignore_stdin_cannot_prompt_password(self, httpbin):
        r = http('--ignore-stdin', '--auth=no-password', httpbin.url + '/get',
                 error_exit_ok=True)
        assert r.exit_status == ExitStatus.ERROR
        assert 'because --ignore-stdin' in r.stderr


# def test_rst_file_syntax(filename):
#     p = subprocess.Popen(
#         ['rst2pseudoxml.py', '--report=1', '--exit-status=1', filename],
#         stderr=subprocess.PIPE,
#         stdout=subprocess.PIPE
#     )
#     err = p.communicate()[1]
#     assert p.returncode == 0, err.decode('utf8')

def test_POST_JSON_data(httpbin_both):
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


def test_error_response_exits_0_without_check_status(httpbin):
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERRO' in r
    assert r.exit_status == ExitStatus.OK
    assert not r.stderr


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


def test_timeout_exit_status(httpbin):

    r = http('--timeout=0.01', 'GET', httpbin.url + '/delay/0.02',
             error_exit_ok=True)
    assert r.exit_status == ExitStatus.ERROR_TIMEOUT


class TestBinaryRequestData:

    def test_binary_stdin(self, httpbin):
        with open(BIN_FILE_PATH, 'rb') as stdin:
            env = TestEnvironment(
                stdin=stdin,
                stdin_isatty=False,
                stdout_isatty=False
            )
            r = http('--print=B', 'POST', httpbin.url + '/post', env=env)
            assert r == BIN_FILE_CONTENT

    def test_binary_file_path(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--print=B', 'POST', httpbin.url + '/post',
                 '@' + BIN_FILE_PATH_ARG, env=env, )
        assert r == BIN_FILE_CONTENT

    def test_binary_file_form(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--print=B', '--form', 'POST', httpbin.url + '/post',
                 'test@' + BIN_FILE_PATH_ARG, env=env)
        assert bytes(BIN_FILE_CONTENT) in bytes(r)


def test_POST_form(httpbin_both):
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert '"foo": "bar"' in r


@pytest.mark.skipif(is_windows,
                    reason='Pretty redirect not supported under Windows')
def test_pretty_redirected_stream(httpbin):
    """Test that --stream works with prettified redirected output."""
    with open(BIN_FILE_PATH, 'rb') as f:
        env = TestEnvironment(colors=256, stdin=f,
                              stdin_isatty=False,
                              stdout_isatty=False)
        r = http('--verbose', '--pretty=all', '--stream', 'GET',
                 httpbin.url + '/get', env=env)
    assert BINARY_SUPPRESSED_NOTICE.decode() in r


def test_follow_all_redirects_shown(httpbin):
    r = http('--follow', '--all', httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 3
    assert r.count('HTTP/1.1 302 FOUND', 2)
    assert HTTP_OK in r


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

def test_headers_empty_value_with_value_gives_error(httpbin):
    with pytest.raises(ParseError):
        http('GET', httpbin + '/headers', 'Accept;SYNTAX_ERROR')


class TestBinaryRequestData:

    def test_binary_stdin(self, httpbin):
        with open(BIN_FILE_PATH, 'rb') as stdin:
            env = TestEnvironment(
                stdin=stdin,
                stdin_isatty=False,
                stdout_isatty=False
            )
            r = http('--print=B', 'POST', httpbin.url + '/post', env=env)
            assert r == BIN_FILE_CONTENT

    def test_binary_file_path(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--print=B', 'POST', httpbin.url + '/post',
                 '@' + BIN_FILE_PATH_ARG, env=env, )
        assert r == BIN_FILE_CONTENT

    def test_binary_file_form(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--print=B', '--form', 'POST', httpbin.url + '/post',
                 'test@' + BIN_FILE_PATH_ARG, env=env)
        assert bytes(BIN_FILE_CONTENT) in bytes(r)


def test_unicode_url_query_arg_item_verbose(httpbin):
    r = http('--verbose', httpbin.url + '/get', u'test==%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


def test_auth_plugin_parse_auth_false(httpbin):

    class Plugin(AuthPlugin):
        auth_type = 'test-parse-false'
        auth_parse = False

        def get_auth(self, username=None, password=None):
            assert username is None
            assert password is None
            assert self.raw_auth == BASIC_AUTH_HEADER_VALUE
            return basic_auth(self.raw_auth)

    plugin_manager.register(Plugin)
    try:
        r = http(
            httpbin + BASIC_AUTH_URL,
            '--auth-type',
            Plugin.auth_type,
            '--auth',
            BASIC_AUTH_HEADER_VALUE,
        )
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)


class TestBinaryResponseData:
    url = 'http://www.google.com/favicon.ico'

    @property
    def bindata(self):
        if not hasattr(self, '_bindata'):
            self._bindata = urlopen(self.url).read()
        return self._bindata

    def test_binary_suppresses_when_terminal(self):
        r = http('GET', self.url)
        assert BINARY_SUPPRESSED_NOTICE.decode() in r

    def test_binary_suppresses_when_not_terminal_but_pretty(self):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--pretty=all', 'GET', self.url,
                 env=env)
        assert BINARY_SUPPRESSED_NOTICE.decode() in r

    def test_binary_included_and_correct_when_suitable(self):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('GET', self.url, env=env)
        assert r == self.bindata

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

def test_keyboard_interrupt_in_program_exit_status(httpbin):
    with mock.patch('httpie.core.program',
                    side_effect=KeyboardInterrupt()):
        r = http('GET', httpbin.url + '/get', error_exit_ok=True)
        assert r.exit_status == ExitStatus.ERROR_CTRL_C


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

class TestLocalhostShorthand:
    def test_expand_localhost_shorthand(self):
        args = parser.parse_args(args=[':'], env=TestEnvironment())
        assert args.url == 'http://localhost'

    def test_expand_localhost_shorthand_with_slash(self):
        args = parser.parse_args(args=[':/'], env=TestEnvironment())
        assert args.url == 'http://localhost/'

    def test_expand_localhost_shorthand_with_port(self):
        args = parser.parse_args(args=[':3000'], env=TestEnvironment())
        assert args.url == 'http://localhost:3000'

    def test_expand_localhost_shorthand_with_path(self):
        args = parser.parse_args(args=[':/path'], env=TestEnvironment())
        assert args.url == 'http://localhost/path'

    def test_expand_localhost_shorthand_with_port_and_slash(self):
        args = parser.parse_args(args=[':3000/'], env=TestEnvironment())
        assert args.url == 'http://localhost:3000/'

    def test_expand_localhost_shorthand_with_port_and_path(self):
        args = parser.parse_args(args=[':3000/path'], env=TestEnvironment())
        assert args.url == 'http://localhost:3000/path'

    def test_dont_expand_shorthand_ipv6_as_shorthand(self):
        args = parser.parse_args(args=['::1'], env=TestEnvironment())
        assert args.url == 'http://::1'

    def test_dont_expand_longer_ipv6_as_shorthand(self):
        args = parser.parse_args(
            args=['::ffff:c000:0280'],
            env=TestEnvironment()
        )
        assert args.url == 'http://::ffff:c000:0280'

    def test_dont_expand_full_ipv6_as_shorthand(self):
        args = parser.parse_args(
            args=['0000:0000:0000:0000:0000:0000:0000:0001'],
            env=TestEnvironment()
        )
        assert args.url == 'http://0000:0000:0000:0000:0000:0000:0000:0001'


def test_error_response_exits_0_without_check_status(httpbin):
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERRO' in r
    assert r.exit_status == ExitStatus.OK
    assert not r.stderr

