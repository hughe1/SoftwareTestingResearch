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


def test_debug():
    r = http('--debug')
    assert r.exit_status == httpie.ExitStatus.OK
    assert 'HTTPie %s' % httpie.__version__ in r.stderr


def test_follow_all_output_options_used_for_redirects(httpbin):
    r = http('--check-status',
             '--follow',
             '--all',
             '--print=H',
             httpbin.url + '/redirect/2')
    assert r.count('GET /') == 3
    assert HTTP_OK not in r


def test_headers_empty_value(httpbin_both):
    r = http('GET', httpbin_both + '/headers')
    assert r.json['headers']['Accept']  # default Accept has value

    r = http('GET', httpbin_both + '/headers', 'Accept;')
    assert r.json['headers']['Accept'] == ''   # Accept has no value


@pytest.mark.skipif(not has_docutils(), reason='docutils not installed')
@pytest.mark.parametrize('filename', filenames)
class TestColors:

    @pytest.mark.parametrize(
        argnames=['mime', 'explicit_json', 'body', 'expected_lexer_name'],
        argvalues=[
            ('application/json',     False, None, 'JSON'),
            ('application/json+foo', False, None, 'JSON'),
            ('application/foo+json', False, None, 'JSON'),
            ('application/json-foo', False, None, 'JSON'),
            ('application/x-json',   False, None, 'JSON'),
            ('foo/json',             False, None, 'JSON'),
            ('foo/json+bar',         False, None, 'JSON'),
            ('foo/bar+json',         False, None, 'JSON'),
            ('foo/json-foo',         False, None, 'JSON'),
            ('foo/x-json',           False, None, 'JSON'),
            ('application/vnd.comverge.grid+hal+json', False, None, 'JSON'),
            ('text/plain',           True, '{}', 'JSON'),
            ('text/plain',           True, 'foo', 'Text only'),
        ]
    )
    def test_get_lexer(self, mime, explicit_json, body, expected_lexer_name):
        lexer = get_lexer(mime, body=body, explicit_json=explicit_json)
        assert lexer is not None
        assert lexer.name == expected_lexer_name

    def test_get_lexer_not_found(self):
        assert get_lexer('xxx/yyy') is None


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


def test_max_redirects(httpbin):
    r = http('--max-redirects=1', '--follow', httpbin.url + '/redirect/3',
             error_exit_ok=True)
    assert r.exit_status == ExitStatus.ERROR_TOO_MANY_REDIRECTS

def test_unicode_url_query_arg_item_verbose(httpbin):
    r = http('--verbose', httpbin.url + '/get', u'test==%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


def test_unicode_form_item_verbose(httpbin):
    r = http('--verbose', '--form',
             'POST', httpbin.url + '/post', u'test=%s' % UNICODE)
    assert HTTP_OK in r
    assert UNICODE in r


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

@mock.patch('httpie.core.get_response')
def test_timeout(get_response):
    def error(msg, *args, **kwargs):
        global error_msg
        error_msg = msg % args

    exc = Timeout('Request timed out')
    exc.request = Request(method='GET', url='http://www.google.com')
    get_response.side_effect = exc
    ret = main(['--ignore-stdin', 'www.google.com'], custom_log_error=error)
    assert ret == ExitStatus.ERROR_TIMEOUT
    assert error_msg == 'Request timed out (30s).'

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


def basic_auth(header=BASIC_AUTH_HEADER_VALUE):

    def inner(r):
        r.headers['Authorization'] = header
        return r

    return inner


def test_unicode_raw_json_item(httpbin):
    r = http('--json', 'POST', httpbin.url + '/post',
             u'test:={ "%s" : [ "%s" ] }' % (UNICODE, UNICODE))
    assert HTTP_OK in r
    assert r.json['json'] == {'test': {UNICODE: [UNICODE]}}


def test_credentials_in_url_auth_flag_has_priority(httpbin_both):
    """When credentials are passed in URL and via -a at the same time,
     then the ones from -a are used."""
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:wrong')
    r = http('--auth=user:password', 'GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_POST_form(httpbin_both):
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert '"foo": "bar"' in r


@pytest.mark.parametrize('follow_flag', ['--follow', '-F'])
def test_follow_without_all_redirects_hidden(httpbin, follow_flag):
    r = http(follow_flag, httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 1
    assert HTTP_OK in r


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


def test_error_response_exits_0_without_check_status(httpbin):
    r = http('GET', httpbin.url + '/status/500')
    assert '500 INTERNAL SERVER ERRO' in r
    assert r.exit_status == ExitStatus.OK
    assert not r.stderr


def test_auth_plugin_require_auth_false_and_auth_provided(httpbin):

    class Plugin(AuthPlugin):
        auth_type = 'test-require-false-yet-provided'
        auth_require = False

        def get_auth(self, username=None, password=None):
            assert self.raw_auth == USERNAME + SEP_CREDENTIALS + PASSWORD
            assert username == USERNAME
            assert password == PASSWORD
            return basic_auth()

    plugin_manager.register(Plugin)
    try:
        r = http(
            httpbin + BASIC_AUTH_URL,
            '--auth-type',
            Plugin.auth_type,
            '--auth',
            USERNAME + SEP_CREDENTIALS + PASSWORD,
        )
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)


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


def test_rst_file_syntax(filename):
    p = subprocess.Popen(
        ['rst2pseudoxml.py', '--report=1', '--exit-status=1', filename],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE
    )
    err = p.communicate()[1]
    assert p.returncode == 0, err.decode('utf8')

def test_follow_all_output_options_used_for_redirects(httpbin):
    r = http('--check-status',
             '--follow',
             '--all',
             '--print=H',
             httpbin.url + '/redirect/2')
    assert r.count('GET /') == 3
    assert HTTP_OK not in r


def test_unicode_digest_auth(httpbin):
    # it doesn't really authenticate us because httpbin
    # doesn't interpret the utf8-encoded auth
    http('--auth-type=digest',
         '--auth', u'test:%s' % UNICODE,
         httpbin.url + u'/digest-auth/auth/test/' + UNICODE)

def test_credentials_in_url_auth_flag_has_priority(httpbin_both):
    """When credentials are passed in URL and via -a at the same time,
     then the ones from -a are used."""
    url = add_auth(httpbin_both.url + '/basic-auth/user/password',
                   auth='user:wrong')
    r = http('--auth=user:password', 'GET', url)
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def test_POST_form(httpbin_both):
    r = http('--form', 'POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert '"foo": "bar"' in r


def has_docutils():
    try:
        # noinspection PyUnresolvedReferences
        import docutils
        return True
    except ImportError:
        return False


def test_unicode_basic_auth(httpbin):
    # it doesn't really authenticate us because httpbin
    # doesn't interpret the utf8-encoded auth
    http('--verbose', '--auth', u'test:%s' % UNICODE,
         httpbin.url + u'/basic-auth/test/' + UNICODE)

