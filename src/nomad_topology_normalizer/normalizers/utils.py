from typing import Iterable
import requests
import time
import re
import random


PUB_CHEM_PUG_PATH = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound'
CAS_API_PATH = 'https://commonchemistry.cas.org/api'
EXTERNAL_API_TIMEOUT = 5


def throttle_wait():
    """Function for waiting before an API request to prevent throttling."""
    time.sleep(random.randint(1, 3))


def pub_chem_add_throttle_header(response: requests.Response, message: str = '') -> str:
    """Function for adding the PubChem PUG API throttling control header to a message."""
    if 'X-Throttling-Control' in response.headers:
        message += f' (Throttling-Control: {response.headers["X-Throttling-Control"]})'
    return message


def pub_chem_api_get_properties(
    cid: int, properties: Iterable[str]
) -> requests.Response:
    """
    Function for performing a get request to the PubChem PUG API to get properties for a
    given compound identifier.

    Args:
        cid (int): The compound identifier of the compound of interest.
        properties (Iterable[str]): The properties to retrieve the value for.

    Returns:
        requests.Response: The response as returned from the PubChem PUG API.
    """
    return requests.get(
        url=f'{PUB_CHEM_PUG_PATH}/cid/{cid}/property/{str.join(",", properties)}/JSON',
        timeout=EXTERNAL_API_TIMEOUT,
    )


def pub_chem_api_get_synonyms(cid: int) -> requests.Response:
    """
    Function for performing a get request to the PubChem PUG API to get properties for a
    given compound identifier.

    Args:
        cid (int): The compound identifier of the compound of interest.

    Returns:
        requests.Response: The response as returned from the PubChem PUG API.
    """
    return requests.get(
        url=f'{PUB_CHEM_PUG_PATH}/cid/{cid}/synonyms/JSON',
        timeout=EXTERNAL_API_TIMEOUT,
    )


def pub_chem_api_search(path: str, search: str) -> requests.Response:
    """
    Function for performing a get request to the PubChem PUG API to search the given path
    for a given string.

    Args:
        path (str): The path (property) to search for.
        search (str): The string to search for a match with.

    Returns:
        requests.Response: The response as returned from the PubChem PUG API.
    """
    return requests.get(
        url=f'{PUB_CHEM_PUG_PATH}/{path}/{search}/cids/JSON',
        timeout=EXTERNAL_API_TIMEOUT,
    )


def cas_api_search(search: str) -> requests.Response:
    """
    Function for performing a get request to the CAS API to search for a match with the
    given string.

    Args:
        search (str): The string to search for a match with.

    Returns:
        requests.Response: The response as returned from the CAS API.
    """
    return requests.get(
        f'{CAS_API_PATH}/search?q={search}',
        timeout=EXTERNAL_API_TIMEOUT,
    )


def cas_api_details(cas_rn: str) -> requests.Response:
    """
    Function for performing a get request to the CAS API to get the details for the
    substance with the given CAS registry number.

    Args:
        cas_rn (str): The CAS registry number of the substance for which to get details.

    Returns:
        requests.Response: The response as returned from the CAS API.
    """
    return requests.get(
        f'{CAS_API_PATH}/detail?cas_rn={cas_rn}',
        timeout=EXTERNAL_API_TIMEOUT,
    )


def is_cas_rn(candidate: str) -> bool:
    """
    Help function for checking if a candidate string is a valid CAS Registry Number.

    Args:
        candidate (str): The candidate string to be checked.

    Returns:
        bool: Whether or not the candidate string is a valid CAS Registry Number.
    """
    try:
        match = re.fullmatch(
            r'(?P<p1>\d{2,7})-(?P<p2>\d{2})-(?P<check>\d{1})', candidate
        )
        check = (
            sum(
                [
                    int(c) * (i + 1)
                    for i, c in enumerate(
                        reversed(match.group('p1') + match.group('p2'))
                    )
                ]
            )
            % 10
        )
        return int(match.group('check')) == check
    except (AttributeError, TypeError):
        return False
