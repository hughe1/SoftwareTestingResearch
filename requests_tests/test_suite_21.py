# -*- coding: utf-8 -*-

"""Tests for Requests."""

from __future__ import division
import json
import os
import pickle
import collections
import contextlib
import warnings

import io
import requests
import pytest
from requests.adapters import HTTPAdapter
from requests.auth import HTTPDigestAuth, _basic_auth_str
from requests.compat import (
    Morsel, cookielib, getproxies, str, urlparse,
    builtin_str, OrderedDict)
from requests.cookies import (
    cookiejar_from_dict, morsel_to_cookie)
from requests.exceptions import (
    ConnectionError, ConnectTimeout, InvalidSchema, InvalidURL,
    MissingSchema, ReadTimeout, Timeout, RetryError, TooManyRedirects,
    ProxyError, InvalidHeader, UnrewindableBodyError, SSLError)
from requests.models import PreparedRequest
from requests.structures import CaseInsensitiveDict
from requests.sessions import SessionRedirectMixin
from requests.models import urlencode
from requests.hooks import default_hooks

# from .compat import StringIO, u
# from .utils import override_environ
from urllib3.util import Timeout as Urllib3Timeout

# Requests to this URL should always fail with a connection timeout (nothing
# listening on that port)
TARPIT = 'http://10.255.255.1'

try:
    from ssl import SSLContext
    del SSLContext
    HAS_MODERN_SSL = True
except ImportError:
    HAS_MODERN_SSL = False

try:
    requests.pyopenssl
    HAS_PYOPENSSL = True
except AttributeError:
    HAS_PYOPENSSL = False

# -*- coding: utf-8 -*-

import threading
import socket
import time

import pytest
import requests
from testserver.server import Server

# -*- coding: utf-8 -*-

import pytest

from requests.structures import CaseInsensitiveDict, LookupDict

# -*- coding: utf-8 -*-

import os
import copy
from io import BytesIO

import pytest
from requests import compat
from requests.cookies import RequestsCookieJar
from requests.structures import CaseInsensitiveDict
from requests.utils import (
    address_in_network, dotted_netmask,
    get_auth_from_url, get_encoding_from_headers,
    get_encodings_from_content, get_environ_proxies,
    guess_filename, guess_json_utf, is_ipv4_address,
    is_valid_cidr, iter_slices, parse_dict_header,
    parse_header_links, prepend_scheme_if_needed,
    requote_uri, select_proxy, should_bypass_proxies, super_len,
    to_key_val_list, to_native_string,
    unquote_header_value, unquote_unreserved,
    urldefragauth, add_dict_to_cookiejar, set_environ)
from requests._internal_utils import unicode_is_ascii


# -*- encoding: utf-8

import sys

import pytest

from requests.help import info


def test_redirect_rfc1808_to_non_ascii_location():
    path = u'š'
    expected_path = b'%C5%A1'
    redirect_request = []  # stores the second request to the server

    def redirect_resp_handler(sock):
        consume_socket_content(sock, timeout=0.5)
        location = u'//{0}:{1}/{2}'.format(host, port, path)
        sock.send(
            b'HTTP/1.1 301 Moved Permanently\r\n'
            b'Content-Length: 0\r\n'
            b'Location: ' + location.encode('utf8') + b'\r\n'
            b'\r\n'
        )
        redirect_request.append(consume_socket_content(sock, timeout=0.5))
        sock.send(b'HTTP/1.1 200 OK\r\n\r\n')

    close_server = threading.Event()
    server = Server(redirect_resp_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = u'http://{0}:{1}'.format(host, port)
        r = requests.get(url=url, allow_redirects=True)
        assert r.status_code == 200
        assert len(r.history) == 1
        assert r.history[0].status_code == 301
        assert redirect_request[0].startswith(b'GET /' + expected_path + b' HTTP/1.1')
        assert r.url == u'{0}/{1}'.format(url, expected_path.decode('ascii'))

        close_server.set()

class TestAddressInNetwork:

    def test_valid(self):
        assert address_in_network('192.168.1.1', '192.168.1.0/24')

    def test_invalid(self):
        assert not address_in_network('172.16.0.1', '192.168.1.0/24')


class TestContentEncodingDetection:

    def test_none(self):
        encodings = get_encodings_from_content('')
        assert not len(encodings)

    @pytest.mark.parametrize(
        'content', (
            # HTML5 meta charset attribute
            '<meta charset="UTF-8">',
            # HTML4 pragma directive
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">',
            # XHTML 1.x served with text/html MIME type
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />',
            # XHTML 1.x served as XML
            '<?xml version="1.0" encoding="UTF-8"?>',
        ))
    def test_pragmas(self, content):
        encodings = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == 'UTF-8'

    def test_precedence(self):
        content = '''
        <?xml version="1.0" encoding="XML"?>
        <meta charset="HTML5">
        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />
        '''.strip()
        assert get_encodings_from_content(content) == ['HTML5', 'HTML4', 'XML']


class TestContentEncodingDetection:

    def test_none(self):
        encodings = get_encodings_from_content('')
        assert not len(encodings)

    @pytest.mark.parametrize(
        'content', (
            # HTML5 meta charset attribute
            '<meta charset="UTF-8">',
            # HTML4 pragma directive
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8">',
            # XHTML 1.x served with text/html MIME type
            '<meta http-equiv="Content-type" content="text/html;charset=UTF-8" />',
            # XHTML 1.x served as XML
            '<?xml version="1.0" encoding="UTF-8"?>',
        ))
    def test_pragmas(self, content):
        encodings = get_encodings_from_content(content)
        assert len(encodings) == 1
        assert encodings[0] == 'UTF-8'

    def test_precedence(self):
        content = '''
        <?xml version="1.0" encoding="XML"?>
        <meta charset="HTML5">
        <meta http-equiv="Content-type" content="text/html;charset=HTML4" />
        '''.strip()
        assert get_encodings_from_content(content) == ['HTML5', 'HTML4', 'XML']


def test_digestauth_401_only_sent_once():
    """Ensure we correctly respond to a 401 challenge once, and then
    stop responding if challenged again.
    """
    text_401 = (b'HTTP/1.1 401 UNAUTHORIZED\r\n'
                b'Content-Length: 0\r\n'
                b'WWW-Authenticate: Digest nonce="6bf5d6e4da1ce66918800195d6b9130d"'
                b', opaque="372825293d1c26955496c80ed6426e9e", '
                b'realm="me@kennethreitz.com", qop=auth\r\n\r\n')

    expected_digest = (b'Authorization: Digest username="user", '
                       b'realm="me@kennethreitz.com", '
                       b'nonce="6bf5d6e4da1ce66918800195d6b9130d", uri="/"')

    auth = requests.auth.HTTPDigestAuth('user', 'pass')

    def digest_failed_response_handler(sock):
        # Respond to initial GET with a challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content.startswith(b"GET / HTTP/1.1")
        sock.send(text_401)

        # Verify we receive an Authorization header in response, then
        # challenge again.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert expected_digest in request_content
        sock.send(text_401)

        # Verify the client didn't respond to second challenge.
        request_content = consume_socket_content(sock, timeout=0.5)
        assert request_content == b''

        return request_content

    close_server = threading.Event()
    server = Server(digest_failed_response_handler, wait_to_close_event=close_server)

    with server as (host, port):
        url = 'http://{0}:{1}/'.format(host, port)
        r = requests.get(url, auth=auth)
        # Verify server didn't authenticate us.
        assert r.status_code == 401
        assert r.history[0].status_code == 401
        close_server.set()

