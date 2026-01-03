"""Download and process ORCID in bulk."""

from __future__ import annotations

import csv
import gzip
import json
import logging
import tarfile
import typing
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import pystow
import ssslm
from curies import NamedReference
from lxml import etree
from pydantic import BaseModel, Field
from pydantic_extra_types.country import CountryAlpha2, _index_by_alpha2
from pystow.utils import safe_open_writer
from semantic_pydantic import SemanticField
from tqdm.auto import tqdm

from .name_utils import clean_name, reconcile_aliases
from .standardize import standardize_role, write_role_counters

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionInfo:
    """A tuple containing information for downloading ORCID data dumps.

    Search on https://orcid.figshare.com/search?q=ORCID+Public+Data+File&sortBy=posted_date&sortType=desc
    for the newest one.
    """

    version: str
    url: str
    fname: str
    size: int
    output_directory_name: str = "output"

    @property
    def raw_module(self) -> pystow.Module:
        """Get the raw module."""
        return pystow.module("orcid", self.version)

    @property
    def output_module(self) -> pystow.Module:
        """Get the output module."""
        return self.raw_module.module(self.output_directory_name)


#: See https://orcid.figshare.com/articles/dataset/ORCID_Public_Data_File_2023/24204912/1
#: and download the "summaries" file. Skip all the activities files
VERSION_2023 = VersionInfo(
    version="2023",
    url="https://orcid.figshare.com/ndownloader/files/42479943",
    fname="ORCID_2023_10_summaries.tar.gz",
    size=18_600_000,
)

#: See https://orcid.figshare.com/articles/dataset/ORCID_Public_Data_File_2024/27151305
#: and download the "summaries" file. Skip all the activities files
VERSION_2024 = VersionInfo(
    version="2024",
    url="https://orcid.figshare.com/ndownloader/files/49560102",
    fname="ORCID_2024_10_summaries.tar.gz",
    size=18_600_000,  # FIXME
)

VERSION_DEFAULT = VERSION_2023

NAMESPACES = {
    "personal-details": "http://www.orcid.org/ns/personal-details",
    "common": "http://www.orcid.org/ns/common",
    "other-name": "http://www.orcid.org/ns/other-name",
    "employment": "http://www.orcid.org/ns/employment",
    "invited-position": "http://www.orcid.org/ns/invited-position",
    "activities": "http://www.orcid.org/ns/activities",
    "education": "http://www.orcid.org/ns/education",
    "external-identifier": "http://www.orcid.org/ns/external-identifier",
    "researcher-url": "http://www.orcid.org/ns/researcher-url",
    "email": "http://www.orcid.org/ns/email",
    "keyword": "http://www.orcid.org/ns/keyword",
    "membership": "http://www.orcid.org/ns/membership",
    "address": "http://www.orcid.org/ns/address",
    "preferences": "http://www.orcid.org/ns/preferences",
}


def _get_raw_module(*, version_info: VersionInfo | None = None) -> pystow.Module:
    if version_info is None:
        version_info = VERSION_DEFAULT
    return version_info.raw_module


def _get_output_module(version_info: VersionInfo | None = None) -> pystow.Module:
    if version_info is None:
        version_info = VERSION_DEFAULT
    return version_info.output_module


def _norm_key(id_type):
    return id_type.lower().replace(" ", "").rstrip(":")


EXTERNAL_ID_SKIP = {
    "iAuthor": "Reuses orcid",
    "中国科学家在线": "iAuthor, but site is dead",
    "JRIN": "Reuses orcid",
    "ORCID": "redundant",
    "ORCID id": "redundant",
    "eScientist": "Reuses orcid",
    "UNE Researcher ID": "not an id",
    "UOW Scholars": "not an id",
    "US EPA VIVO": "not an id",
    "Chalmers ID": "not an id",
    "HKUST Profile": "not an id",
    "Custom": "garb",
    "Profile system identifier": "garb",
    "CTI Vitae": "dead website",
    "Pitt ID": "dead website",
    "VIVO Cornell": "dead website",
    "Technical University of Denmark CWIS": "dead website",
    "HKU ResearcherPage": "dead website",
    "Digital Author ID": "DAI is not specific service",
    "Digital Author ID (DAI)": "DAI is not specific service",
    "dai": "DAI is not specific service",
}
EXTERNAL_ID_SKIP = {_norm_key(k): v for k, v in EXTERNAL_ID_SKIP.items()}
#: Mapping from ORCID keys to Bioregistry prefixes for external IDs
EXTERNAL_ID_MAPPING = {
    "ResearcherID": "wos.researcher",
    "RID": "wos.researcher",
    "Web of Science Researcher ID": "wos.researcher",
    "other-id - Web of Science": "wos.researcher",
    "Scopus Author ID": "scopus",
    "Scopus ID": "scopus",
    "ID de autor de Scopus": "scopus",
    "???person.personsources.scopusauthor???": "scopus",
    "Loop profile": "loop",
    "github": "github",
    "ISNI": "isni",
    "Google Scholar": "google.scholar",
    "gnd": "gnd",
    "Authenticus": "authenticus",
    "AuthenticusID": "authenticus",
    "AuthID": "authenticus",
    "ID Dialnet": "dialnet.author",
    "Dialnet ID": "dialnet.author",
    "SciProfiles": "sciprofiles",
    "Sciprofile": "sciprofiles",
    "Ciência ID": "cienciavitae",
    "KAKEN": "kaken",
    "Researcher Name Resolver ID": "kaken",
    "SSRN": "ssrn.author",
    "socialscienceresearchnetwork": "ssrn.author",
    "ssrnauthorpage": "ssrn.author",
    "ssrnpage": "ssrn.author",
}

EXTERNAL_ID_MAPPING = {_norm_key(k): v for k, v in EXTERNAL_ID_MAPPING.items()}
UNMAPPED_EXTERNAL_ID: set[str] = set()
PERSONAL_KEYS = {
    "website",
    "homepage",
    "blog",
    "personalpage",
    "personalhomepage",
    "personalwebsite",
    "personalwebsites",
    "personalweb-page",
    "personalwebpage",
    "webpage",
    "personal",
    "professionalwebsite",
    "personalsite",
    "personalblog",
    "mywebsite",
    "mysite",
    "officialweb-page",
    "sitiowebpersonal",
    "paginaweb",
    "personelwebsite",
    "blogpessoal",
    "mypersonalsite",
    "mypersonalblog",
    "personalweb-site",
    "web-site",
    "professionalblog",
    "personalwebsiteandblog",
    "myweb",
    "homewebsite",
    "personalweb",
    "mypersonalwebsite",
    "blogpersonal",
}


def ensure_summaries(*, version_info: VersionInfo) -> Path:
    """Ensure the ORCID summaries file (32+ GB) is downloaded."""
    return _get_raw_module(version_info=version_info).ensure(
        url=version_info.url, name=version_info.fname
    )


class Work(BaseModel):
    """A model representing a creative work."""

    pubmed: str = Field(..., title="PubMed identifier")


class Date(BaseModel):
    """A model representing a date."""

    year: int
    month: int | None = None
    day: int | None = None


class Affiliation(BaseModel):
    """A model representing an affiliation (either education or employment)."""

    name: str = None
    start: Date | None = Field(None, title="Start Year")
    end: Date | None = Field(None, title="End Year")
    role: None | str | NamedReference = None
    country: str
    department: str = None
    #xrefs: dict[str, str] = Field(default_factory=dict, title="Database Cross-references")

    # xrefs includes ror, ringgold, grid, funderregistry, lei
    # LEI see https://www.gleif.org/en/lei-data/gleif-concatenated-file/download-the-concatenated-file

    @property
    def ror(self) -> str | None:
        """Get the affiliation's ROR identifier, if available."""
        return self.xrefs.get("ror")


class Record(BaseModel):
    """A model representing a person."""

    orcid: str = SemanticField(..., prefix="orcid")
    name: str = Field(..., min_length=1)
    homepage: str | None = Field(None)
    locale: str | None = Field(None)
    countries: list[CountryAlpha2] = Field(
        default_factory=list, description="The ISO 3166-1 alpha-2 country codes (uppercase)"
    )
    aliases: list[str] = Field(default_factory=list)
    xrefs: dict[str, str] = Field(default_factory=dict, title="Database Cross-references")
    works: list[Work] = Field(default_factory=list)
    employments: list[Affiliation] = Field(default_factory=list)
    educations: list[Affiliation] = Field(default_factory=list)
    invited_positions: list[Affiliation] = Field(default_factory=list)
    memberships: list[Affiliation] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    commons_image: str | None = None

    @property
    def commons_image_url(self) -> str | None:
        """Get the Wikimedia Commons image URL, if available."""
        if self.commons_image:
            return f"http://commons.wikimedia.org/wiki/Special:FilePath/{self.commons_image}"
        return None

    def is_high_quality(self) -> bool:
        """Return if the record is high quality."""
        if not self.name:
            return False
        # just see if there's literally anything in there
        return bool(
            any("ror" in employment.xrefs for employment in self.employments)
            or any("ror" in education.xrefs for education in self.educations)
            # or any("ror" in membership.xrefs for membership in self.memberships)
            or self.works
            or self.xrefs
        )

    @property
    def email(self) -> str | None:
        """Get the first email, if available."""
        return self.emails[0] if self.emails else None

    @property
    def country(self) -> str | None:
        """Get the first country, if available."""
        return self.countries[0] if self.countries else None

    @property
    def github(self) -> str | None:
        """Get the researcher's GitHub username, if available."""
        return self.xrefs.get("github")

    @property
    def linkedin(self) -> str | None:
        """Get the researcher's LinkedIn username, if available."""
        return self.xrefs.get("linkedin")

    @property
    def loop(self) -> str | None:
        """Get the researcher's Loop identifier, if available."""
        return self.xrefs.get("loop")

    @property
    def wos(self) -> str | None:
        """Get the researcher's Web of Science identifier, if available."""
        return self.xrefs.get("wos.researcher")

    @property
    def dblp(self) -> str | None:
        """Get the researcher's DBLP identifier, if available."""
        return self.xrefs.get("dblp")

    @property
    def scopus(self) -> str | None:
        """Get the researcher's Scopus identifier, if available."""
        return self.xrefs.get("scopus")

    @property
    def google(self) -> str | None:
        """Get the researcher's Google Scholar identifier, if available."""
        return self.xrefs.get("google.scholar")

    @property
    def wikidata(self) -> str | None:
        """Get the researcher's Wikidata identifier, if available."""
        return self.xrefs.get("wikidata")

    @property
    def mastodon(self) -> str | None:
        """Get the researcher's Mastodon handle, if available."""
        return self.xrefs.get("mastodon")

    @property
    def current_affiliation_ror(self) -> str | None:
        """Guess the current affiliation and return its ROR identifier, if available."""
        # assume that if there are employments listed that are not over yet,
        # then these surpass education
        for employment in self.employments:
            if employment.ror and employment.end is None:
                return employment.ror

        for education in self.educations:
            if education.ror and education.end is None:
                return education.ror

        return None


def iter_tarfile_members(path: Path):
    tar_file = tarfile.open(path)
    while member := tar_file.next():
        if not member.name.endswith(".xml"):
            continue
        yield tar_file.extractfile(member)
    tar_file.close()


def get_records(
    *, force: bool = False, version_info: VersionInfo | None = None
) -> dict[str, Record]:
    """Parse ORCID summary XML files, takes about an hour."""
    return {record.orcid: record for record in iter_records(force=force, version_info=version_info)}


def process_file(  # noqa:C901
    file,
    orcid_to_wikidata: dict[str, str],
    orcid_to_wikimedia_commons: dict[str, str],
) -> Record | None:
    """Process a file obnect for an XML file.

    :param file: An XML file object
    :param orcid_to_wikidata: A one-to-one mapping from ORCID to Wikidata identifiers
    :param orcid_to_wikimedia_commons: A mapping from ORCID to Wikimedia Commons image tags
    :return: A record

    .. code-block:: python

    """
    tree = etree.parse(file)

    orcid = tree.findtext(".//common:path", namespaces=NAMESPACES)
    if not orcid:
        return None

    family_name = tree.findtext(".//personal-details:family-name", namespaces=NAMESPACES)
    if family_name is None:
        family_name = ''
    given_names = tree.findtext(".//personal-details:given-names", namespaces=NAMESPACES)
    if given_names is None:
        given_names = ''
    if family_name or given_names:
        label_name = f"{given_names.strip()} {family_name.strip()}"
    else:
        label_name = None

    credit_name = tree.findtext(".//personal-details:credit-name", namespaces=NAMESPACES)
    if credit_name:
        credit_name = credit_name.strip()

    if not credit_name and not label_name:
        # Skip records that don't have any kinds of labels
        return None

    aliases: set[str] = set()
    if not credit_name:
        name = label_name
    else:
        name = credit_name
        if label_name is not None:
            aliases.add(label_name)

    name = name and clean_name(name)
    aliases.update(_iter_other_names(tree))
    if name in aliases:  # make sure there's no duplicate
        aliases.remove(name)
    name, aliases = reconcile_aliases(name, aliases)
    # despite best efforts, no names are possible
    if name is None:
        return None

    record: dict[str, Any] = {"orcid": orcid, "name": name}
    if aliases:
        record["aliases"] = sorted(aliases)

    employments = _get_employments(tree)
    if employments:
        record["employments"] = employments

    educations = _get_educations(tree)
    if educations:
        record["educations"] = educations

    invited = _get_invited_positions(tree)
    if educations:
        record["invited_positions"] = invited

    # memberships = _get_memberships(tree)
    # if memberships:
    #     record["memberships"] = memberships

    # ids, homepage = _get_external_identifiers(tree, orcid=orcid)
    # if wikidata_id := orcid_to_wikidata.get(orcid):
    #     ids["wikidata"] = wikidata_id
    # if ids:
    #     record["xrefs"] = ids
    # if homepage:
    #     record["homepage"] = homepage
    # if image := orcid_to_wikimedia_commons.get(orcid):
    #     record["commons_image"] = image

    # if works := _get_works(tree, orcid=orcid):
    #     record["works"] = works

    if emails := _get_emails(tree):
        record["emails"] = emails

    if keywords := _get_keywords(tree):
        record["keywords"] = sorted(keywords)

    # if countries := _get_countries(tree, orcid=orcid):
    #     record["countries"] = countries
    if locale := _get_locale(tree, orcid=orcid):
        record["locale"] = locale

    return Record.model_validate(record)


def _iter_other_names(t) -> Iterable[str]:
    for part in t.findall(".//other-name:content", namespaces=NAMESPACES):
        part = part.text.strip()
        for z in part.split(";"):
            z = z.strip()
            if z is not None and " " in z and len(z) < 60:
                yield clean_name(z.strip())


UNKNOWN_SOURCES = {}
LOWERCASE_THESE_SOURCES = {"RINGGOLD", "GRID", "LEI"}
UNKNOWN_NAMES: typing.Counter[str] = Counter()
UNKNOWN_NAMES_EXAMPLES: dict[str, str] = {}
UNKNOWN_NAMES_FULL: dict[str, str] = {}


def _get_emails(tree) -> list[str]:
    return [
        email.text.strip()
        for email in tree.findall(".//email:emails/email:email/email:email", namespaces=NAMESPACES)
    ]


def _get_keywords(tree) -> Iterable[str]:
    return [
        keyword.text.strip()
        for keyword in tree.findall(
            ".//keyword:keywords/keyword:keyword/keyword:content", namespaces=NAMESPACES
        )
        if keyword.text
    ]


def _get_countries(tree, orcid) -> list[str]:
    rv = []
    for country in tree.findall(
        ".//address:addresses/address:address/address:country", namespaces=NAMESPACES
    ):
        value = country.text
        if not value:
            continue
        value = value.strip().upper()
        if value == "XK":
            # XK is a proposed code for Kosovo, but isn't valid.
            # Only an issue for a few dozen records
            continue
        elif value not in _index_by_alpha2():
            tqdm.write(f"[{orcid}] invalid 2 letter country code: {value}")
            continue
        rv.append(value)
    return rv


def _get_locale(tree, orcid) -> str | None:
    value = tree.findtext(".//preferences:preferences/preferences:locale", namespaces=NAMESPACES)
    if value is None:
        return None
    return value.strip()


def _get_works(tree, orcid) -> list[dict[str, str]]:
    # get a subset of all works with pubmed IDs. TODO extend to other IDs
    pmids = set()
    for g in tree.findall(
        ".//activities:works/activities:group/common:external-ids", namespaces=NAMESPACES
    ):
        if g.findtext(".//common:external-id-type", namespaces=NAMESPACES) == "pmid":
            value: str | None = g.findtext(".//common:external-id-value", namespaces=NAMESPACES)
            if not value:
                continue
            value_std = _standardize_pubmed(value)
            if not value_std:
                continue
            if not value_std.isnumeric():
                tqdm.write(f"[{orcid}] unstandardized PubMed: '{value}'")
                continue
            pmids.add(value_std)
    return [{"pubmed": pmid} for pmid in sorted(pmids)]


PUBMED_PREFIXES = [
    "http://www.ncbi.nlm.nih.gov/pubmed/",
    "https://www.ncbi.nlm.nih.gov/pubmed/",
    "https://www-ncbi-nlm-nih-gov.proxy.bib.ucl.ac.be:2443/pubmed/",
    "http://europepmc.org/abstract/med/",
    "https://pubmed.ncbi.nlm.nih.gov/",
    "www.ncbi.nlm.nih.gov/pubmed/",
    "PMID: ",
    "PMID:",
    "PubMed PMID: ",
    "MEDLINE:",
    "[PMID: ",
    "PubMed:",
    "PubMed ID: ",
    "ncbi.nlm.nih.gov/pubmed/",
    "PMid:",
    "PubMed ",
    "PMID",
    "PMID ",
]


def _standardize_pubmed(pubmed: str) -> str | None:
    """Standardize a pubmed field.

    :param pubmed: A string that might somehow represent a pubmed identifier
    :returns: A cleaned pubmed identifier, if possible

    2023 statistics:

    - correct: 3,175,196 (99.85%)
    - needs processing: 2,832 (0.09%)
    - junk: 2,080 (0.07%)

    what was in here? a mashup of:

    - DOIs
    - PMC identifiers,
    - a few stray strings that contain a combination of pubmed, PMC,
    - a lot with random text (keywords)
    - some with full text citations
    """
    pubmed = pubmed.strip().strip(".").rstrip("/").strip()
    if pubmed.isnumeric():
        return pubmed
    for x in PUBMED_PREFIXES:
        if pubmed.startswith(x):
            parts = pubmed.removeprefix(x).strip().split()
            if parts:
                return parts[0]
    if pubmed.endswith("E7"):
        pubmed = str(int(float(pubmed)))
        return pubmed
    return None


def _get_employments(tree):
    elements = tree.findall(".//employment:employment-summary", namespaces=NAMESPACES)
    return _get_affiliations(elements)


def _get_educations(tree):
    elements = tree.findall(
        ".//education:education-summary", namespaces=NAMESPACES
    )
    return _get_affiliations(elements)

def _get_invited_positions(tree):
    elements = tree.findall(
        ".//invited-position:invited-position-summary", namespaces=NAMESPACES
    )
    return _get_affiliations(elements)


def _get_memberships(tree):
    elements = tree.findall(
        ".//activities:memberships//membership:membership-summary", namespaces=NAMESPACES
    )
    return _get_affiliations(elements)


def _get_affiliations(elements):
    results = []
    for element in elements:
        if element is None:
            continue
        organization_element = element.find(".//common:organization", namespaces=NAMESPACES)
        if organization_element is None:
            continue

        name = organization_element.findtext(".//common:name", namespaces=NAMESPACES)
        # if not name:
        #     continue
        record = {"name": name.strip()}

        if (start_date := element.find(".//common:start-date", namespaces=NAMESPACES)) is not None:
            record["start"] = _get_date(start_date)
        if (end_date := element.find(".//common:end-date", namespaces=NAMESPACES)) is not None:
            record["end"] = _get_date(end_date)

        if role := _get_role(element):
            record["role"] = role

        if (country := element.find(".//common:country", namespaces=NAMESPACES)) is not None:
            record["country"] = element.findtext(".//common:country", namespaces=NAMESPACES)

        if (country := element.find(".//common:department-name", namespaces=NAMESPACES)) is not None:
            record["department"] = element.findtext(".//common:department-name", namespaces=NAMESPACES)
        results.append(Affiliation.model_validate(record))
    return results


def _get_date(date_element) -> Date | None:
    year = date_element.findtext(".//common:year", namespaces=NAMESPACES)
    if year is None:
        return None
    month = date_element.findtext(".//common:month", namespaces=NAMESPACES)
    day = date_element.findtext(".//common:day", namespaces=NAMESPACES)
    return Date(year=year, month=month, day=day)


def _get_disambiguated_organization(
    organization_element, name
) -> dict[str, str]:
    references = {}
    for de in organization_element.findall(
        ".//common:disambiguated-organization", namespaces=NAMESPACES
    ):
        source = de.findtext(".//common:disambiguation-source", namespaces=NAMESPACES)
        link = de.findtext(".//common:disambiguated-organization-identifier", namespaces=NAMESPACES)
        if not link:
            continue
        link = link.strip()
        if source == "ROR":
            references["ror"] = link.removeprefix("https://ror.org/")
        elif source in LOWERCASE_THESE_SOURCES:
            references[source.lower()] = link
        elif source == "FUNDREF":
            references["funderregistry"] = link.removeprefix("http://dx.doi.org/10.13039/")
        elif source not in UNKNOWN_SOURCES:
            tqdm.write(f"unhandled source: {source} / link: {link}")
            UNKNOWN_SOURCES[source] = link
    return references


#: Role text needs to be longer than this
MINIMUM_ROLE_LENGTH = 2


def _get_role(element) -> NamedReference | str | None:
    text = element.findtext(".//common:role-title", namespaces=NAMESPACES)
    if not text:
        return None
    return text
    # label, _, reference =text, _, #  standardize_role(text)
    # if len(label) < MINIMUM_ROLE_LENGTH:
    #     return None
    # if reference:
    #     return reference
    # return label


def ground_researcher_unambiguous(name: str) -> str | None:
    """Ground a name based on ORCID names/aliases."""
    matches = ground_researcher(name)
    if len(matches) != 1:
        return None
    return matches[0].identifier


def write_schema(path: Path) -> None:
    """Write the JSON schema."""
    schema = Record.model_json_schema()
    path.write_text(json.dumps(schema, indent=2))
