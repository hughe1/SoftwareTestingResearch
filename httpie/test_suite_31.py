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


def test_default_options(httpbin):
    env = TestEnvironment()
    env.config['default_options'] = ['--form']
    env.config.save()
    r = http(httpbin.url + '/post', 'foo=bar', env=env)
    assert r.json['form'] == {"foo": "bar"}


class TestSessionFlow(SessionTestBase):
    """
    These tests start with an existing session created in `setup_method()`.

    """

    def start_session(self, httpbin):
        """
        Start a full-blown session with a custom request header,
        authorization, and response cookies.

        """
        super(TestSessionFlow, self).start_session(httpbin)
        r1 = http('--follow', '--session=test', '--auth=username:password',
                  'GET', httpbin.url + '/cookies/set?hello=world',
                  'Hello:World',
                  env=self.env())
        assert HTTP_OK in r1

    def test_session_created_and_reused(self, httpbin):
        self.start_session(httpbin)
        # Verify that the session created in setup_method() has been used.
        r2 = http('--session=test',
                  'GET', httpbin.url + '/get', env=self.env())
        assert HTTP_OK in r2
        assert r2.json['headers']['Hello'] == 'World'
        assert r2.json['headers']['Cookie'] == 'hello=world'
        assert 'Basic ' in r2.json['headers']['Authorization']

    def test_session_update(self, httpbin):
        self.start_session(httpbin)
        # Get a response to a request from the original session.
        r2 = http('--session=test', 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r2

        # Make a request modifying the session data.
        r3 = http('--follow', '--session=test', '--auth=username:password2',
                  'GET', httpbin.url + '/cookies/set?hello=world2',
                  'Hello:World2',
                  env=self.env())
        assert HTTP_OK in r3

        # Get a response to a request from the updated session.
        r4 = http('--session=test', 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r4
        assert r4.json['headers']['Hello'] == 'World2'
        assert r4.json['headers']['Cookie'] == 'hello=world2'
        assert (r2.json['headers']['Authorization'] !=
                r4.json['headers']['Authorization'])

    def test_session_read_only(self, httpbin):
        self.start_session(httpbin)
        # Get a response from the original session.
        r2 = http('--session=test', 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r2

        # Make a request modifying the session data but
        # with --session-read-only.
        r3 = http('--follow', '--session-read-only=test',
                  '--auth=username:password2', 'GET',
                  httpbin.url + '/cookies/set?hello=world2', 'Hello:World2',
                  env=self.env())
        assert HTTP_OK in r3

        # Get a response from the updated session.
        r4 = http('--session=test', 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r4

        # Origin can differ on Travis.
        del r2.json['origin'], r4.json['origin']
        # Different for each request.

        # Should be the same as before r3.
        assert r2.json == r4.json


class TestQuerystring:
    def test_query_string_params_in_url(self, httpbin):
        r = http('--print=Hhb', 'GET', httpbin.url + '/get?a=1&b=2')
        path = '/get?a=1&b=2'
        url = httpbin.url + path
        assert HTTP_OK in r
        assert 'GET %s HTTP/1.1' % path in r
        assert '"url": "%s"' % url in r

    def test_query_string_params_items(self, httpbin):
        r = http('--print=Hhb', 'GET', httpbin.url + '/get', 'a==1')
        path = '/get?a=1'
        url = httpbin.url + path
        assert HTTP_OK in r
        assert 'GET %s HTTP/1.1' % path in r
        assert '"url": "%s"' % url in r

    def test_query_string_params_in_url_and_items_with_duplicates(self,
                                                                  httpbin):
        r = http('--print=Hhb', 'GET',
                 httpbin.url + '/get?a=1&a=1', 'a==1', 'a==1')
        path = '/get?a=1&a=1&a=1&a=1'
        url = httpbin.url + path
        assert HTTP_OK in r
        assert 'GET %s HTTP/1.1' % path in r
        assert '"url": "%s"' % url in r


@pytest.mark.skipif(is_windows, reason='Unix-only')
def test_output_devnull(httpbin):
    """
    https://github.com/jakubroztocil/httpie/issues/252

    """
    http('--output=/dev/null', httpbin + '/get')

class TestSession(SessionTestBase):
    """Stand-alone session tests."""

    def test_session_ignored_header_prefixes(self, httpbin):
        self.start_session(httpbin)
        r1 = http('--session=test', 'GET', httpbin.url + '/get',
                  'Content-Type: text/plain',
                  'If-Unmodified-Since: Sat, 29 Oct 1994 19:43:31 GMT',
                  env=self.env())
        assert HTTP_OK in r1
        r2 = http('--session=test', 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r2
        assert 'Content-Type' not in r2.json['headers']
        assert 'If-Unmodified-Since' not in r2.json['headers']

    def test_session_by_path(self, httpbin):
        self.start_session(httpbin)
        session_path = os.path.join(self.config_dir, 'session-by-path.json')
        r1 = http('--session=' + session_path, 'GET', httpbin.url + '/get',
                  'Foo:Bar', env=self.env())
        assert HTTP_OK in r1

        r2 = http('--session=' + session_path, 'GET', httpbin.url + '/get',
                  env=self.env())
        assert HTTP_OK in r2
        assert r2.json['headers']['Foo'] == 'Bar'

    @pytest.mark.skipif(
        sys.version_info >= (3,),
        reason="This test fails intermittently on Python 3 - "
               "see https://github.com/jakubroztocil/httpie/issues/282")
    def test_session_unicode(self, httpbin):
        self.start_session(httpbin)

        r1 = http('--session=test', u'--auth=test:' + UNICODE,
                  'GET', httpbin.url + '/get', u'Test:%s' % UNICODE,
                  env=self.env())
        assert HTTP_OK in r1

        r2 = http('--session=test', '--verbose', 'GET',
                  httpbin.url + '/get', env=self.env())
        assert HTTP_OK in r2

        # FIXME: Authorization *sometimes* is not present on Python3
        assert (r2.json['headers']['Authorization'] ==
                HTTPBasicAuth.make_header(u'test', UNICODE))
        # httpbin doesn't interpret utf8 headers
        assert UNICODE in r2

    def test_session_default_header_value_overwritten(self, httpbin):
        self.start_session(httpbin)
        # https://github.com/jakubroztocil/httpie/issues/180
        r1 = http('--session=test',
                  httpbin.url + '/headers', 'User-Agent:custom',
                  env=self.env())
        assert HTTP_OK in r1
        assert r1.json['headers']['User-Agent'] == 'custom'

        r2 = http('--session=test', httpbin.url + '/headers', env=self.env())
        assert HTTP_OK in r2
        assert r2.json['headers']['User-Agent'] == 'custom'

    def test_download_in_session(self, httpbin):
        # https://github.com/jakubroztocil/httpie/issues/412
        self.start_session(httpbin)
        cwd = os.getcwd()
        os.chdir(gettempdir())
        try:
            http('--session=test', '--download',
                 httpbin.url + '/get', env=self.env())
        finally:
            os.chdir(cwd)

def test_unicode_form_item(httpbin):
    r = http('--form', 'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['form'] == {'test': UNICODE}


@mock.patch('httpie.core.get_response')
def test_error_traceback(get_response):
    exc = ConnectionError('Connection aborted')
    exc.request = Request(method='GET', url='http://www.google.com')
    get_response.side_effect = exc
    with raises(ConnectionError):
        main(['--ignore-stdin', '--traceback', 'www.google.com'])


def test_DELETE(httpbin_both):
    r = http('DELETE', httpbin_both + '/delete')
    assert HTTP_OK in r


def test_version():
    r = http('--version', error_exit_ok=True)
    assert r.exit_status == httpie.ExitStatus.OK
    # FIXME: py3 has version in stdout, py2 in stderr
    assert httpie.__version__ == r.stderr.strip() + r.strip()


@mock.patch('httpie.input.AuthCredentials._getpass',
            new=lambda self, prompt: 'password')
def test_password_prompt(httpbin):
    r = http('--auth', 'user',
             'GET', httpbin.url + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_missing_auth(httpbin):
    r = http(
        '--auth-type=basic',
        'GET',
        httpbin + '/basic-auth/user/password',
        error_exit_ok=True
    )
    assert HTTP_OK not in r
    assert '--auth required' in r.stderr

def test_POST_form(httpbin_both):
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert '"foo": "bar"' in r


class TestDownloads:
    # TODO: more tests

    def test_actual_download(self, httpbin_both, httpbin):
        robots_txt = '/robots.txt'
        body = urlopen(httpbin + robots_txt).read().decode()
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--download', httpbin_both.url + robots_txt, env=env)
        assert 'Downloading' in r.stderr
        assert '[K' in r.stderr
        assert 'Done' in r.stderr
        assert body == r

    def test_download_with_Content_Length(self, httpbin_both):
        devnull = open(os.devnull, 'w')
        downloader = Downloader(output_file=devnull, progress_file=devnull)
        downloader.start(Response(
            url=httpbin_both.url + '/',
            headers={'Content-Length': 10}
        ))
        time.sleep(1.1)
        downloader.chunk_downloaded(b'12345')
        time.sleep(1.1)
        downloader.chunk_downloaded(b'12345')
        downloader.finish()
        assert not downloader.interrupted

    def test_download_no_Content_Length(self, httpbin_both):
        devnull = open(os.devnull, 'w')
        downloader = Downloader(output_file=devnull, progress_file=devnull)
        downloader.start(Response(url=httpbin_both.url + '/'))
        time.sleep(1.1)
        downloader.chunk_downloaded(b'12345')
        downloader.finish()
        assert not downloader.interrupted

    def test_download_interrupted(self, httpbin_both):
        devnull = open(os.devnull, 'w')
        downloader = Downloader(output_file=devnull, progress_file=devnull)
        downloader.start(Response(
            url=httpbin_both.url + '/',
            headers={'Content-Length': 5}
        ))
        downloader.chunk_downloaded(b'1234')
        downloader.finish()
        assert downloader.interrupted

class TestItemParsing:

    key_value = KeyValueArgType(*input.SEP_GROUP_ALL_ITEMS)

    def test_invalid_items(self):
        items = ['no-separator']
        for item in items:
            pytest.raises(argparse.ArgumentTypeError, self.key_value, item)

    def test_escape_separator(self):
        items = input.parse_items([
            # headers
            self.key_value(r'foo\:bar:baz'),
            self.key_value(r'jack\@jill:hill'),

            # data
            self.key_value(r'baz\=bar=foo'),

            # files
            self.key_value(r'bar\@baz@%s' % FILE_PATH_ARG),
        ])
        # `requests.structures.CaseInsensitiveDict` => `dict`
        headers = dict(items.headers._store.values())

        assert headers == {
            'foo:bar': 'baz',
            'jack@jill': 'hill',
        }
        assert items.data == {'baz=bar': 'foo'}
        assert 'bar@baz' in items.files

    @pytest.mark.parametrize(('string', 'key', 'sep', 'value'), [
        ('path=c:\windows', 'path', '=', 'c:\windows'),
        ('path=c:\windows\\', 'path', '=', 'c:\windows\\'),
        ('path\==c:\windows', 'path=', '=', 'c:\windows'),
    ])
    def test_backslash_before_non_special_character_does_not_escape(
            self, string, key, sep, value):
        expected = KeyValue(orig=string, key=key, sep=sep, value=value)
        actual = self.key_value(string)
        assert actual == expected

    def test_escape_longsep(self):
        items = input.parse_items([
            self.key_value(r'bob\:==foo'),
        ])
        assert items.params == {'bob:': 'foo'}

    def test_valid_items(self):
        items = input.parse_items([
            self.key_value('string=value'),
            self.key_value('Header:value'),
            self.key_value('Unset-Header:'),
            self.key_value('Empty-Header;'),
            self.key_value('list:=["a", 1, {}, false]'),
            self.key_value('obj:={"a": "b"}'),
            self.key_value('ed='),
            self.key_value('bool:=true'),
            self.key_value('file@' + FILE_PATH_ARG),
            self.key_value('query==value'),
            self.key_value('string-embed=@' + FILE_PATH_ARG),
            self.key_value('raw-json-embed:=@' + JSON_FILE_PATH_ARG),
        ])

        # Parsed headers
        # `requests.structures.CaseInsensitiveDict` => `dict`
        headers = dict(items.headers._store.values())
        assert headers == {
            'Header': 'value',
            'Unset-Header': None,
            'Empty-Header': ''
        }

        # Parsed data
        raw_json_embed = items.data.pop('raw-json-embed')
        assert raw_json_embed == json.loads(JSON_FILE_CONTENT)
        items.data['string-embed'] = items.data['string-embed'].strip()
        assert dict(items.data) == {
            "ed": "",
            "string": "value",
            "bool": True,
            "list": ["a", 1, {}, False],
            "obj": {"a": "b"},
            "string-embed": FILE_CONTENT,
        }

        # Parsed query string parameters
        assert items.params == {'query': 'value'}

        # Parsed file fields
        assert 'file' in items.files
        assert (items.files['file'][1].read().strip().
                decode('utf8') == FILE_CONTENT)

    def test_multiple_file_fields_with_same_field_name(self):
        items = input.parse_items([
            self.key_value('file_field@' + FILE_PATH_ARG),
            self.key_value('file_field@' + FILE_PATH_ARG),
        ])
        assert len(items.files['file_field']) == 2

    def test_multiple_text_fields_with_same_field_name(self):
        items = input.parse_items(
            [self.key_value('text_field=a'),
             self.key_value('text_field=b')],
            data_class=DataDict
        )
        assert items.data['text_field'] == ['a', 'b']
        assert list(items.data.items()) == [
            ('text_field', 'a'),
            ('text_field', 'b'),
        ]


def test_3xx_check_status_redirects_allowed_exits_0(httpbin):
    r = http('--check-status', '--follow',
             'GET', httpbin.url + '/status/301',
             error_exit_ok=True)
    # The redirect will be followed so 200 is expected.
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.OK


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


def test_GET(httpbin_both):
    r = http('GET', httpbin_both + '/get')
    assert HTTP_OK in r


def test_3xx_check_status_redirects_allowed_exits_0(httpbin):
    r = http('--check-status', '--follow',
             'GET', httpbin.url + '/status/301',
             error_exit_ok=True)
    # The redirect will be followed so 200 is expected.
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.OK


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

def test_follow_all_redirects_shown(httpbin):
    r = http('--follow', '--all', httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 3
    assert r.count('HTTP/1.1 302 FOUND', 2)
    assert HTTP_OK in r


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

def test_default_options(httpbin):
    env = TestEnvironment()
    env.config['default_options'] = ['--form']
    env.config.save()
    r = http(httpbin.url + '/post', 'foo=bar', env=env)
    assert r.json['form'] == {"foo": "bar"}


def test_4xx_check_status_exits_4(httpbin):
    r = http('--check-status', 'GET', httpbin.url + '/status/401',
             error_exit_ok=True)
    assert '401 UNAUTHORIZED' in r
    assert r.exit_status == ExitStatus.ERROR_HTTP_4XX
    # Also stderr should be empty since stdout isn't redirected.
    assert not r.stderr


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

def test_version():
    r = http('--version', error_exit_ok=True)
    assert r.exit_status == httpie.ExitStatus.OK
    # FIXME: py3 has version in stdout, py2 in stderr
    assert httpie.__version__ == r.stderr.strip() + r.strip()


@pytest.mark.skipif(not is_windows, reason='windows-only')
class TestWindowsOnly:

    @pytest.mark.skipif(True,
                        reason='this test for some reason kills the process')
    def test_windows_colorized_output(self, httpbin):
        # Spits out the colorized output.
        http(httpbin.url + '/get', env=Environment())


def test_timeout_exit_status(httpbin):

    r = http('--timeout=0.01', 'GET', httpbin.url + '/delay/0.02',
             error_exit_ok=True)
    assert r.exit_status == ExitStatus.ERROR_TIMEOUT


def test_missing_auth(httpbin):
    r = http(
        '--auth-type=basic',
        'GET',
        httpbin + '/basic-auth/user/password',
        error_exit_ok=True
    )
    assert HTTP_OK not in r
    assert '--auth required' in r.stderr

def basic_auth(header=BASIC_AUTH_HEADER_VALUE):

    def inner(r):
        r.headers['Authorization'] = header
        return r

    return inner


def test_missing_auth(httpbin):
    r = http(
        '--auth-type=basic',
        'GET',
        httpbin + '/basic-auth/user/password',
        error_exit_ok=True
    )
    assert HTTP_OK not in r
    assert '--auth required' in r.stderr
