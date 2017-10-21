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

    def test_GET_with_data_auto_JSON_headers(self, httpbin):
        # JSON headers should automatically be set also for GET with data.
        r = http('POST', httpbin.url + '/post', 'a=b')
        assert HTTP_OK in r
        assert r.json['headers']['Accept'] == JSON_ACCEPT
        assert r.json['headers']['Content-Type'] == 'application/json'

    def test_POST_explicit_JSON_auto_JSON_accept(self, httpbin):
        r = http('--json', 'POST', httpbin.url + '/post')
        assert HTTP_OK in r
        assert r.json['headers']['Accept'] == JSON_ACCEPT
        # Make sure Content-Type gets set even with no data.
        # https://github.com/jakubroztocil/httpie/issues/137
        assert 'application/json' in r.json['headers']['Content-Type']

    def test_GET_explicit_JSON_explicit_headers(self, httpbin):
        r = http('--json', 'GET', httpbin.url + '/headers',
                 'Accept:application/xml',
                 'Content-Type:application/xml')
        assert HTTP_OK in r
        assert '"Accept": "application/xml"' in r
        assert '"Content-Type": "application/xml"' in r

    def test_POST_form_auto_Content_Type(self, httpbin):
        r = http('--form', 'POST', httpbin.url + '/post')
        assert HTTP_OK in r
        assert '"Content-Type": "application/x-www-form-urlencoded' in r

    def test_POST_form_Content_Type_override(self, httpbin):
        r = http('--form', 'POST', httpbin.url + '/post',
                 'Content-Type:application/xml')
        assert HTTP_OK in r
        assert '"Content-Type": "application/xml"' in r

    def test_print_only_body_when_stdout_redirected_by_default(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('GET', httpbin.url + '/get', env=env)
        assert 'HTTP/' not in r

    def test_print_overridable_when_stdout_redirected(self, httpbin):
        env = TestEnvironment(stdin_isatty=True, stdout_isatty=False)
        r = http('--print=h', 'GET', httpbin.url + '/get', env=env)
        assert HTTP_OK in r

@pytest.mark.parametrize('argument_name', ['--auth-type', '-A'])
def test_digest_auth(httpbin_both, argument_name):
    r = http(argument_name + '=digest', '--auth=user:password',
             'GET', httpbin_both.url + '/digest-auth/auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


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

class TestNoOptions:

    def test_valid_no_options(self, httpbin):
        r = http('--verbose', '--no-verbose', 'GET', httpbin.url + '/get')
        assert 'GET /get HTTP/1.1' not in r

    def test_invalid_no_options(self, httpbin):
        r = http('--no-war', 'GET', httpbin.url + '/get',
                 error_exit_ok=True)
        assert r.exit_status == 1
        assert 'unrecognized arguments: --no-war' in r.stderr
        assert 'GET /get HTTP/1.1' not in r


def test_PUT(httpbin_both):
    r = http('PUT', httpbin_both + '/put', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


def test_unicode_url(httpbin):
    r = http(httpbin.url + u'/get?test=' + UNICODE)
    assert HTTP_OK in r
    assert r.json['args'] == {'test': UNICODE}

# def test_unicode_url_verbose(self):
#     r = http(httpbin.url + '--verbose', u'/get?test=' + UNICODE)
#     assert HTTP_OK in r


def test_credentials_in_url(httpbin_both):
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:password')
    r = http('GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_credentials_in_url_auth_flag_has_priority(httpbin_both):
    """When credentials are passed in URL and via -a at the same time,
     then the ones from -a are used."""
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:wrong')
    r = http('--auth=user:password', 'GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


class TestNoOptions:

    def test_valid_no_options(self, httpbin):
        r = http('--verbose', '--no-verbose', 'GET', httpbin.url + '/get')
        assert 'GET /get HTTP/1.1' not in r

    def test_invalid_no_options(self, httpbin):
        r = http('--no-war', 'GET', httpbin.url + '/get',
                 error_exit_ok=True)
        assert r.exit_status == 1
        assert 'unrecognized arguments: --no-war' in r.stderr
        assert 'GET /get HTTP/1.1' not in r


def test_unicode_json_item(httpbin):
    r = http('--json', 'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert r.json['json'] == {'test': UNICODE}


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


def test_3xx_check_status_redirects_allowed_exits_0(httpbin):
    r = http('--check-status', '--follow',
             'GET', httpbin.url + '/status/301',
             error_exit_ok=True)
    # The redirect will be followed so 200 is expected.
    assert HTTP_OK in r
    assert r.exit_status == ExitStatus.OK


def test_credentials_in_url(httpbin_both):
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:password')
    r = http('GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


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


@pytest.mark.skipif(not is_windows, reason='windows-only')
class TestWindowsOnly:

    @pytest.mark.skipif(True,
                        reason='this test for some reason kills the process')
    def test_windows_colorized_output(self, httpbin):
        # Spits out the colorized output.
        http(httpbin.url + '/get', env=Environment())


# @pytest.mark.skipif(not has_docutils(), reason='docutils not installed')
# @pytest.mark.parametrize('filename', filenames)
def test_POST_JSON_data(httpbin_both):
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


def test_credentials_in_url(httpbin_both):
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:password')
    r = http('GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def rst_filenames():
    for root, dirnames, filenames in os.walk(os.path.dirname(TESTS_ROOT)):
        if '.tox' not in root:
            for filename in fnmatch.filter(filenames, '*.rst'):
                yield os.path.join(root, filename)


filenames = list(rst_filenames())
assert filenames


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


def has_docutils():
    try:
        # noinspection PyUnresolvedReferences
        import docutils
        return True
    except ImportError:
        return False


@pytest.mark.skipif(is_windows, reason='Unix-only')
def test_output_devnull(httpbin):
    """
    https://github.com/jakubroztocil/httpie/issues/252

    """
    http('--output=/dev/null', httpbin + '/get')

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


@mock.patch('httpie.input.AuthCredentials._getpass',
            new=lambda self, prompt: 'password')
def test_password_prompt(httpbin):
    r = http('--auth', 'user',
             'GET', httpbin.url + '/basic-auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


@pytest.mark.parametrize('follow_flag', ['--follow', '-F'])
def test_follow_without_all_redirects_hidden(httpbin, follow_flag):
    r = http(follow_flag, httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 1
    assert HTTP_OK in r

