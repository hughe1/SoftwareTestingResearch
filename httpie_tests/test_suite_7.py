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


def test_POST_JSON_data(httpbin_both):
    r = http('POST', httpbin_both + '/post', 'foo=bar')
    assert HTTP_OK in r
    assert r.json['json']['foo'] == 'bar'


@pytest.mark.parametrize('argument_name', ['--auth-type', '-A'])
def test_digest_auth(httpbin_both, argument_name):
    r = http(argument_name + '=digest', '--auth=user:password',
             'GET', httpbin_both.url + '/digest-auth/auth/user/password')
    assert HTTP_OK in r
    assert r.json == {'authenticated': True, 'user': 'user'}


def has_docutils():
    try:
        # noinspection PyUnresolvedReferences
        import docutils
        return True
    except ImportError:
        return False


class TestMultipartFormDataFileUpload:

    def test_non_existent_file_raises_parse_error(self, httpbin):
        with pytest.raises(ParseError):
            http('--form',
                 'POST', httpbin.url + '/post', 'foo@/__does_not_exist__')

