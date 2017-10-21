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


def has_docutils():
    try:
        # noinspection PyUnresolvedReferences
        import docutils
        return True
    except ImportError:
        return False


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


def test_follow_all_redirects_shown(httpbin):
    r = http('--follow', '--all', httpbin.url + '/redirect/2')
    assert r.count('HTTP/1.1') == 3
    assert r.count('HTTP/1.1 302 FOUND', 2)
    assert HTTP_OK in r


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



def rst_filenames():
    for root, dirnames, filenames in os.walk(os.path.dirname(TESTS_ROOT)):
        if '.tox' not in root:
            for filename in fnmatch.filter(filenames, '*.rst'):
                yield os.path.join(root, filename)


filenames = list(rst_filenames())
assert filenames

