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

# -*- coding: utf-8 -*-

import pytest
import threading
import requests

from testserver.server import Server, consume_socket_content



@pytest.mark.skipif(sys.version_info[:2] != (2,6), reason="Only run on Python 2.6")
def test_system_ssl_py26():
    """OPENSSL_VERSION_NUMBER isn't provided in Python 2.6, verify we don't
    blow up in this case.
    """
    assert info()['system_ssl'] == {'version': ''}


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


class TestTestServer:

    def test_basic(self):
        """messages are sent and received properly"""
        question = b"success?"
        answer = b"yeah, success"

        def handler(sock):
            text = sock.recv(1000)
            assert text == question
            sock.sendall(answer)

        with Server(handler) as (host, port):
            sock = socket.socket()
            sock.connect((host, port))
            sock.sendall(question)
            text = sock.recv(1000)
            assert text == answer
            sock.close()

    def test_server_closes(self):
        """the server closes when leaving the context manager"""
        with Server.basic_response_server() as (host, port):
            sock = socket.socket()
            sock.connect((host, port))

            sock.close()

        with pytest.raises(socket.error):
            new_sock = socket.socket()
            new_sock.connect((host, port))

    def test_text_response(self):
        """the text_response_server sends the given text"""
        server = Server.text_response_server(
            "HTTP/1.1 200 OK\r\n" +
            "Content-Length: 6\r\n" +
            "\r\nroflol"
        )

        with server as (host, port):
            r = requests.get('http://{0}:{1}'.format(host, port))

            assert r.status_code == 200
            assert r.text == u'roflol'
            assert r.headers['Content-Length'] == '6'

    def test_basic_response(self):
        """the basic response server returns an empty http response"""
        with Server.basic_response_server() as (host, port):
            r = requests.get('http://{0}:{1}'.format(host, port))
            assert r.status_code == 200
            assert r.text == u''
            assert r.headers['Content-Length'] == '0'

    def test_basic_waiting_server(self):
        """the server waits for the block_server event to be set before closing"""
        block_server = threading.Event()

        with Server.basic_response_server(wait_to_close_event=block_server) as (host, port):
            sock = socket.socket()
            sock.connect((host, port))
            sock.sendall(b'send something')
            time.sleep(2.5)
            sock.sendall(b'still alive')
            block_server.set()  # release server block

    def test_multiple_requests(self):
        """multiple requests can be served"""
        requests_to_handle = 5

        server = Server.basic_response_server(requests_to_handle=requests_to_handle)

        with server as (host, port):
            server_url = 'http://{0}:{1}'.format(host, port)
            for _ in range(requests_to_handle):
                r = requests.get(server_url)
                assert r.status_code == 200

            # the (n+1)th request fails
            with pytest.raises(requests.exceptions.ConnectionError):
                r = requests.get(server_url)

    @pytest.mark.skip(reason="this fails non-deterministically under pytest-xdist")
    def test_request_recovery(self):
        """can check the requests content"""
        # TODO: figure out why this sometimes fails when using pytest-xdist.
        server = Server.basic_response_server(requests_to_handle=2)
        first_request = b'put your hands up in the air'
        second_request = b'put your hand down in the floor'

        with server as address:
            sock1 = socket.socket()
            sock2 = socket.socket()

            sock1.connect(address)
            sock1.sendall(first_request)
            sock1.close()

            sock2.connect(address)
            sock2.sendall(second_request)
            sock2.close()

        assert server.handler_results[0] == first_request
        assert server.handler_results[1] == second_request

    def test_requests_after_timeout_are_not_received(self):
        """the basic response handler times out when receiving requests"""
        server = Server.basic_response_server(request_timeout=1)

        with server as address:
            sock = socket.socket()
            sock.connect(address)
            time.sleep(1.5)
            sock.sendall(b'hehehe, not received')
            sock.close()

        assert server.handler_results[0] == b''

    def test_request_recovery_with_bigger_timeout(self):
        """a biggest timeout can be specified"""
        server = Server.basic_response_server(request_timeout=3)
        data = b'bananadine'

        with server as address:
            sock = socket.socket()
            sock.connect(address)
            time.sleep(1.5)
            sock.sendall(data)
            sock.close()

        assert server.handler_results[0] == data

    def test_server_finishes_on_error(self):
        """the server thread exits even if an exception exits the context manager"""
        server = Server.basic_response_server()
        with pytest.raises(Exception):
            with server:
                raise Exception()

        assert len(server.handler_results) == 0

        # if the server thread fails to finish, the test suite will hang
        # and get killed by the jenkins timeout.

    def test_server_finishes_when_no_connections(self):
        """the server thread exits even if there are no connections"""
        server = Server.basic_response_server()
        with server:
            pass

        assert len(server.handler_results) == 0

        # if the server thread fails to finish, the test suite will hang
        # and get killed by the jenkins timeout.

class TestAddressInNetwork:

    def test_valid(self):
        assert address_in_network('192.168.1.1', '192.168.1.0/24')

    def test_invalid(self):
        assert not address_in_network('172.16.0.1', '192.168.1.0/24')

