import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import tarfile
import textwrap
import time
from datetime import datetime
from email.utils import parsedate_to_datetime
import pyhmmer.plan7
import pyhmmer.hmmer
from tqdm import tqdm
import urllib.request
from platformdirs import user_config_dir
import pandas as pd
import requests
from tqdm import tqdm
from shutil import copyfile

# Define package directory path
package_dir = os.path.dirname(os.path.abspath(__file__))


class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def initialize_config():
    app_name = "Astra"
    app_author = "YourOrg"  # Replace with the actual name of your organization or app author

    # Use os.path.expanduser to get the proper config path
    config_dir = os.path.expanduser("~/.config/Astra")
    default_db_json_path = os.path.join(config_dir, 'hmm_databases.json')

    # Check if hmm_databases.json exists in the user's config directory
    if not os.path.exists(default_db_json_path):
        # Path to the hmm_databases.json file in the repository/package directory
        repo_db_json_path = os.path.join(package_dir, 'hmm_databases.json')
        
        # Check if hmm_databases.json exists in the repository/package directory
        if os.path.exists(repo_db_json_path):
            # Copy hmm_databases.json from the repository/package directory to the user's config directory
            os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
            copyfile(repo_db_json_path, default_db_json_path)
            print(f"'hmm_databases.json' copied to {default_db_json_path}")
        else:
            print("hmm_databases.json not found in the package directory. Please ensure it's included in the repository.")
            sys.exit()

    # Load the hmm_databases.json now that it's ensured to exist
    with open(default_db_json_path, 'r') as f:
        hmm_databases = json.load(f)

    # A user config written by an older Astra won't know about databases added
    # since. Merge in any new entries so they become installable without the
    # user having to blow away their config (and their 'installed' flags).
    repo_db_json_path = os.path.join(package_dir, 'hmm_databases.json')
    if os.path.exists(repo_db_json_path):
        with open(repo_db_json_path, 'r') as f:
            packaged = json.load(f)

        known = {db['name'] for db in hmm_databases['db_urls']}
        new_dbs = [db for db in packaged['db_urls'] if db['name'] not in known]
        if new_dbs:
            hmm_databases['db_urls'].extend(new_dbs)
            with open(default_db_json_path, 'w') as f:
                json.dump(hmm_databases, f, indent=4)
            print("Added newly available databases to your config: "
                  + ', '.join(db['name'] for db in new_dbs))

    return hmm_databases

def load_config():
    app_name = "Astra"
    app_author = "YourOrg"  # Replace with the actual name of your organization or app author

    # Use os.path.expanduser to get the proper config path
    config_dir = os.path.expanduser("~/.config/Astra")
    default_db_json_path = os.path.join(config_dir, 'hmm_databases.json')

    # Attempt to load the existing hmm_databases.json
    if os.path.exists(default_db_json_path):
        with open(default_db_json_path, 'r') as f:
            hmm_databases = json.load(f)
    else:
        print("hmm_databases.json not found!! Please raise an issue on github.")
        sys.exit()

    # If 'db_path' is empty, prompt the user for the directory to store HMM databases
    if not hmm_databases.get('db_path'):
        # Use platformdirs to get the standard configuration directory
        default_db_path = user_config_dir(app_name, app_author)
        print(f"The default directory for HMM databases is: {default_db_path}")
        user_input = input("Would you like to use the default directory for the HMM databases? [Y/n] ").strip().lower()
        if user_input == 'n':
            hmm_databases['db_path'] = input("Please enter the full path to the desired HMM database directory: ")
        else:
            hmm_databases['db_path'] = default_db_path
        
        # Ensure the HMM database directory exists
        os.makedirs(hmm_databases['db_path'], exist_ok=True)

        # Update the hmm_databases.json file with the new db_path
        with open(default_db_json_path, 'w') as f:
            json.dump(hmm_databases, f)
    
    # Always ensure db_path is properly expanded
    hmm_databases['db_path'] = os.path.expanduser(hmm_databases['db_path'])

    return hmm_databases





def show_available_databases(parsed_json):
    print("Available databases:")
    
    # Group by molecule type
    protein_dbs = [db for db in parsed_json['db_urls'] if not db['installed'] and db['molecule_type'] == 'protein']
    nucleotide_dbs = [db for db in parsed_json['db_urls'] if not db['installed'] and db['molecule_type'] == 'nucleotide']
    
    if protein_dbs:
        print("  Protein Databases:")
        for db in protein_dbs:
            print(f"    - {db['name']}")
            if 'notes' in db:
                wrapped_notes = textwrap.fill(db['notes'], initial_indent='      Notes: ', subsequent_indent='            ')
                print(wrapped_notes)
            if 'citation' in db:
                wrapped_citation = textwrap.fill(db['citation'], initial_indent='      Citation: ', subsequent_indent='               ')
                print(wrapped_citation)
            print('\n')
    
    if nucleotide_dbs:
        print("  Nucleotide Databases:")
        for db in nucleotide_dbs:
            print(f"    - {db['name']}")
            if 'notes' in db:
                wrapped_notes = textwrap.fill(db['notes'], initial_indent='      Notes: ', subsequent_indent='            ')
                print(wrapped_notes)
            if 'citation' in db:
                wrapped_citation = textwrap.fill(db['citation'], initial_indent='      Citation: ', subsequent_indent='               ')
                print(wrapped_citation)
            print('\n')
    print("HMM databases requiring licenses: SUPERFAMILY (https://supfam.mrc-lmb.cam.ac.uk/SUPERFAMILY/models.html), PRISM (https://prism.adapsyn.com/)")

    return


def download_progress_hook(count, block_size, total_size):
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r%2d%%" % percent)
    sys.stdout.flush()

def press_hmm_database(db_dir, db_name=None):
    """Concatenate individual HMM files into a single pressed database.

    Creates ``{db_name}.hmm`` (concatenated text) and the four pressed
    index files (``.h3m``, ``.h3i``, ``.h3f``, ``.h3p``) inside *db_dir*.
    Loading from a pressed database is ~50x faster than parsing thousands
    of individual ``.hmm`` files.

    If the directory contains a single multi-model ``.hmm`` file, it is
    pressed in-place (no concatenation needed).
    """
    if db_name is None:
        db_name = os.path.basename(db_dir.rstrip('/'))

    pressed_base = os.path.join(db_dir, db_name)
    # Already pressed?
    if all(os.path.exists(f"{pressed_base}.{ext}") for ext in ('h3m', 'h3i', 'h3f', 'h3p')):
        print(f"  {db_name} is already pressed — skipping.")
        return pressed_base

    hmm_files = sorted(f for f in os.listdir(db_dir)
                       if f.endswith(('.hmm', '.HMM'))
                       and f != f"{db_name}.hmm")  # Don't re-read our own concat file

    if not hmm_files:
        print(f"  No .hmm files found in {db_dir} — nothing to press.")
        return None

    t0 = time.perf_counter()

    if len(hmm_files) == 1:
        # Single multi-model file — press directly
        src = os.path.join(db_dir, hmm_files[0])
        print(f"  Pressing {hmm_files[0]} ({db_name})...")
        with pyhmmer.plan7.HMMFile(src) as hf:
            hmms = list(hf)
    else:
        # Many individual files — read all into memory, then press
        print(f"  Reading {len(hmm_files)} HMM files for {db_name}...")
        hmms = []
        for fname in tqdm(hmm_files, desc="  Loading HMMs"):
            fpath = os.path.join(db_dir, fname)
            with pyhmmer.plan7.HMMFile(fpath) as hf:
                hmms.append(hf.read())

    print(f"  Pressing {len(hmms)} HMMs → {pressed_base}.h3{{m,i,f,p}}...")
    pyhmmer.hmmer.hmmpress(hmms, pressed_base)

    elapsed = time.perf_counter() - t0
    print(f"  Done pressing {db_name} ({elapsed:.1f}s)")
    return pressed_base


def remote_version(url):
    """Release identifier for a download: the source file's Last-Modified date.

    Most of the databases Astra installs publish no version string at all, but
    every HTTP source exposes a modification date, which is enough to tell two
    installs of the same database apart. Returns None if the server won't say.
    """
    try:
        request = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(request) as response:
            last_modified = response.headers.get('Last-Modified')
    except Exception as exc:
        print(f"Could not determine the remote version of {url}: {exc}")
        return None

    if not last_modified:
        return None

    try:
        return parsedate_to_datetime(last_modified).strftime('%Y-%m-%d')
    except (TypeError, ValueError):
        return last_modified


def hmm_build_date(hmm_path):
    """The build date in an HMM's DATE field, as YYYY-MM-DD.

    Lets us identify which release a profile came from when the config has no
    version recorded for it (e.g. it was installed by an older Astra).
    """
    try:
        with open(hmm_path) as handle:
            for line in handle:
                if line.startswith('DATE'):
                    stamp = line[4:].strip()
                    try:
                        return datetime.strptime(stamp, '%a %b %d %H:%M:%S %Y').strftime('%Y-%m-%d')
                    except ValueError:
                        return stamp
                if line.startswith('HMM '):
                    break
    except OSError:
        pass

    return None


def record_version(db_entry, source_url, version=None):
    """Stamp a database entry with what was installed, from where, and when."""
    if version is None and source_url.startswith('http'):
        version = remote_version(source_url)

    db_entry['source_url'] = source_url
    db_entry['version'] = version
    db_entry['installed_date'] = time.strftime('%Y-%m-%d')
    return version


def install_KOFAM():

    #Separate function to install KOFAM because we need to manually add bitscore cutoffs to HMM models
    #which drastically reduces size of the output (288M from a 36M metaproteome vs. Pfam's 24M output!!)
    db_name = 'KOFAM'
    parsed_json = load_config()
    config = load_config()
    db_path = os.path.expandvars(os.path.expanduser(config['db_path']))
    print('DB PATH IN INSTALL_KOFAM: {}'.format(db_path))


    for db in parsed_json['db_urls']:
        #Look. I have to organize this as a for loop because of the JSON structure.
        #I know we're only accessing one element. Leave me alone
        #Feel free to tell me alternatives if you read this far and know what to do! 
        #I'm too lazy to ask GPT. This works just fine

        if (db['installed'] == True and db['installation_dir'] != '') and db['name'] == db_name:
            #Database is already installed; just say so.
            print("Database {} already installed.".format(db['name']))
            continue
        if db['name'] == db_name:
            target_folder = os.path.expanduser(os.path.join(db_path, db_name))
            
            os.makedirs(target_folder, exist_ok=True)
            
            print(f"Downloading {db_name} to {target_folder}...")
            
            # Download the database
            url = db['url']

            file_name = url.split('/')[-1]
            download_path = os.path.join(target_folder, file_name)
            
            if not os.path.exists(download_path):
                # Download the file with progress bar
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:  
                    urllib.request.urlretrieve(url, download_path, reporthook=t.update_to)
            else:
                print("KOFAM profiles already detected in download directory. Decompressing...")
                
            # Extract the file if it's a tar archive
            if tarfile.is_tarfile(download_path):
                print(f"Extracting {file_name}...")
                with tarfile.open(download_path, 'r') as tar_ref:
                    tar_ref.extractall(target_folder)
                os.remove(download_path)  # Remove the original tar file
                
                # Check if a single directory was extracted
                extracted_files = os.listdir(target_folder)
                if len(extracted_files) == 1 and os.path.isdir(os.path.join(target_folder, extracted_files[0])):
                    single_dir = os.path.join(target_folder, extracted_files[0])
                    for file_to_move in os.listdir(single_dir):
                        shutil.move(os.path.join(single_dir, file_to_move), target_folder)
                    os.rmdir(single_dir)  # Remove the now-empty directory
            

            # Mark installation as complete in hmm_databases.json
            db['installed'] = True
            db["installation_dir"] = target_folder  # Add installation directory
            record_version(db, url)
            # Write changes to the JSON file 
            json_path = os.path.join(db_path, 'hmm_databases.json')  # Use db_path here
            print(json_path)
            with open(json_path, 'w') as f:
                json.dump(parsed_json, f, indent=4)
            print(f"{db_name} successfully downloaded and extracted.")

    ####
    #Now we've installed the HMMs, let's grab profiles.gz.
    ####

    print("Downloading KOFAM thresholds list...")
    ko_list_url = 'https://www.genome.jp/ftp/db/kofam/ko_list.gz'
    file_name = ko_list_url.split('/')[-1]
    ko_list_path = os.path.join(db_path, file_name)
    # Download the file with progress bar
    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:  
        urllib.request.urlretrieve(ko_list_url, ko_list_path, reporthook=t.update_to)

    print("Decompressing ko_list...")
    with gzip.open(ko_list_path, 'rb') as f_in:
        with open(ko_list_path[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print("Adding kofam bitscore thresholds to HMM files...")
    #Now parse ko_list as a pandas DF
    ko_list = pd.read_csv(os.path.join(db_path, 'ko_list'), sep='\t')

    # Not all models actually have thresholds; insist that they do
    # All this extraneous code is for the purposes of avoiding a pandas warning. Don't judge me
    # Create a mask for rows where the threshold is not '-'
    mask = ko_list['threshold'] != '-'

    # Use the mask to filter rows and convert the 'threshold' column to float
    ko_list.loc[mask, 'threshold'] = ko_list.loc[mask, 'threshold'].astype(float)


    
    kofam_dir = os.path.join(db_path, 'KOFAM')
    #Iterate only on rows of ko_list as all other HMMs will lack thresholds
    for index, row in ko_list.iterrows():
        threshold = row.threshold
        knum = row.knum
        #Get path to HMM file
        hmm_file = os.path.join(kofam_dir,'{}.hmm'.format(knum))
        #Add the threshold and overwrite the file!!
        add_threshold(hmm_file, threshold)

    # Press the database for fast loading at search time
    print("\nPressing KOFAM database for fast loading...")
    press_hmm_database(kofam_dir, db_name='KOFAM')
    return

def add_threshold(hmm_file_path, threshold):
    # Some thresholds in ko_list are specified for HMMs not provided by the package...

    if not os.path.exists(hmm_file_path) or threshold == 0.0 or threshold == '-':
        # '-' values shouldn't still exist in this set, but if the threshold is 0
        # or the HMM is specified in KO_list but not provided in the HMM set,
        # let's just ignore it and not add bad thresholds
        return

    with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file:
        hmm = hmm_file.read()

    hmm.cutoffs.gathering = threshold, threshold
    hmm.cutoffs.trusted = threshold, threshold
    hmm.cutoffs.noise = threshold, threshold

    with open(hmm_file_path, "wb") as dst:
        hmm.write(dst)

    return


def add_hyddb_thresholds(hmm_file_path, ga_threshold, nc_threshold):
    """
    Add GA (gathering) and NC (noise) thresholds to all HMMs in a file.
    Used for HydDB where conservative and loose thresholds differ.
    """
    if not os.path.exists(hmm_file_path):
        return

    # Read all HMMs from the file
    hmms = []
    with pyhmmer.plan7.HMMFile(hmm_file_path) as hmm_file:
        for hmm in hmm_file:
            hmm.cutoffs.gathering = ga_threshold, ga_threshold
            hmm.cutoffs.noise = nc_threshold, nc_threshold
            hmms.append(hmm)

    # Write all HMMs back to the file
    with open(hmm_file_path, "wb") as dst:
        for hmm in hmms:
            hmm.write(dst)


def install_HydDB():
    """
    Install HydDB hydrogenase HMM profiles with appropriate bitscore thresholds.

    Thresholds from HydDB README (https://github.com/GreeningLab/HydDB):
    - [FeFe]: Conservative (GA)=50, Loose (NC)=15.9
    - [NiFe]: Conservative (GA)=120, Loose (NC)=34.5
    - [Fe]-only: Loose (NC)=54.4 (no conservative threshold available, use loose for GA)
    """
    db_name = 'HydDB'
    parsed_json = load_config()
    config = load_config()
    db_path = os.path.expandvars(os.path.expanduser(config['db_path']))

    # Check if already installed
    for db in parsed_json['db_urls']:
        if db['name'] == db_name:
            if db['installed'] and db['installation_dir']:
                print(f"Database {db_name} already installed.")
                return

    target_folder = os.path.expanduser(os.path.join(db_path, db_name))
    os.makedirs(target_folder, exist_ok=True)

    print(f"Downloading {db_name} to {target_folder}...")

    # HMM files and their thresholds (GA=conservative, NC=loose)
    # Thresholds from: https://github.com/GreeningLab/HydDB/blob/main/README.md
    hmm_files = {
        'FeFe-HydDB_MM2022.hmm': {'GA': 50.0, 'NC': 15.9},
        'NiFe-HydDB_MM2022.hmm': {'GA': 120.0, 'NC': 34.5},
        'Fe_only-HydDB_MM2022.hmm': {'GA': 54.4, 'NC': 54.4},  # No conservative threshold available
    }

    base_url = 'https://raw.githubusercontent.com/GreeningLab/HydDB/main/hmm_profiles/'

    for hmm_file, thresholds in hmm_files.items():
        url = base_url + hmm_file
        download_path = os.path.join(target_folder, hmm_file)

        # Download the file
        print(f"Downloading {hmm_file}...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=hmm_file) as t:
            urllib.request.urlretrieve(url, download_path, reporthook=t.update_to)

        # Add thresholds to all HMMs in the file
        print(f"Adding thresholds to {hmm_file} (GA={thresholds['GA']}, NC={thresholds['NC']})...")
        add_hyddb_thresholds(download_path, thresholds['GA'], thresholds['NC'])

    # Update config
    for db in parsed_json['db_urls']:
        if db['name'] == db_name:
            db['installed'] = True
            db['installation_dir'] = target_folder
            record_version(db, base_url)
            break

    # Write updated config
    config_dir = os.path.expanduser("~/.config/Astra")
    json_path = os.path.join(config_dir, 'hmm_databases.json')
    with open(json_path, 'w') as f:
        json.dump(parsed_json, f, indent=4)

    print(f"{db_name} successfully downloaded and thresholds applied.")

    # Press the database for fast loading at search time
    print(f"\nPressing {db_name} database for fast loading...")
    press_hmm_database(target_folder, db_name=db_name)


def fetch_ko_thresholds(db_path, kos):
    """Bitscore thresholds for a set of KOs, from KOfam's ``ko_list``.

    Downloads ``ko_list`` if it isn't already sitting in ``db_path`` (KOFAM's
    installer leaves it there).  KOs whose threshold is '-' are omitted.
    """
    ko_list_path = os.path.join(db_path, 'ko_list')

    if not os.path.exists(ko_list_path):
        ko_list_url = 'https://www.genome.jp/ftp/db/kofam/ko_list.gz'
        gz_path = ko_list_path + '.gz'
        print("Downloading ko_list for marker thresholds...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc='ko_list.gz') as t:
            urllib.request.urlretrieve(ko_list_url, gz_path, reporthook=t.update_to)
        with gzip.open(gz_path, 'rb') as f_in, open(ko_list_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    ko_list = pd.read_csv(ko_list_path, sep='\t')
    wanted = ko_list[ko_list['knum'].isin(kos) & (ko_list['threshold'] != '-')]
    return {row['knum']: float(row['threshold']) for _, row in wanted.iterrows()}


def install_RP16():
    """Install the 16 ribosomal-protein markers used by ``astra search --16rp``.

    All 16 markers are KOfam KOs, so rather than vendoring profiles into the
    package they are lifted out of the KOFAM database Astra already knows how
    to install.  If KOFAM isn't installed, the profiles are pulled from
    upstream instead; KOfam ships thresholds in ``ko_list`` rather than in the
    profiles themselves, so those get their cutoffs injected here (an
    installed KOFAM has already had this done to it).

    RP16 inherits its version from whichever KOFAM release it was built from,
    which is recorded in the config alongside the profiles.
    """
    from astra import rp16

    db_name = 'RP16'
    parsed_json = load_config()
    db_path = os.path.expandvars(os.path.expanduser(parsed_json['db_path']))

    for db in parsed_json['db_urls']:
        if db['name'] == db_name and db['installed'] and db['installation_dir']:
            print(f"Database {db_name} already installed.")
            return

    target_folder = os.path.join(db_path, db_name)
    os.makedirs(target_folder, exist_ok=True)

    kos = list(rp16.RP16_MARKERS)
    source_version = None

    kofam_dir = None
    for db in parsed_json['db_urls']:
        if db['name'] == 'KOFAM' and db['installed'] and db['installation_dir']:
            if os.path.isdir(db['installation_dir']):
                kofam_dir = db['installation_dir']
                source_version = db.get('version')
            break

    if kofam_dir is not None:
        print(f"Taking {len(kos)} markers from the installed KOFAM database...")
        missing = []
        for ko in kos:
            src = os.path.join(kofam_dir, f"{ko}.hmm")
            if not os.path.exists(src):
                missing.append(ko)
                continue
            shutil.copyfile(src, os.path.join(target_folder, rp16.profile_filename(ko)))
        if missing:
            # Fall back to upstream for whatever the local KOFAM didn't have.
            print(f"Not found in local KOFAM: {', '.join(missing)}. Fetching from upstream...")
            source_version = fetch_kofam_markers(missing, target_folder, db_path, rp16)
    else:
        print(f"KOFAM is not installed; fetching {len(kos)} markers from upstream...")
        source_version = fetch_kofam_markers(kos, target_folder, db_path, rp16)

    expected = {rp16.profile_filename(ko) for ko in kos}
    absent = sorted(f for f in expected if not os.path.exists(os.path.join(target_folder, f)))
    if absent:
        print(f"ERROR: {db_name} installation incomplete, missing: {', '.join(absent)}")
        return

    if source_version is None:
        # A KOFAM installed before Astra tracked versions won't have one
        # recorded, but the profiles themselves carry their build date.
        source_version = hmm_build_date(os.path.join(target_folder, sorted(expected)[0]))

    for index, db in enumerate(parsed_json['db_urls']):
        if db['name'] == db_name:
            parsed_json['db_urls'][index]['installed'] = True
            parsed_json['db_urls'][index]['installation_dir'] = target_folder
            record_version(parsed_json['db_urls'][index],
                           source_url=db['url'], version=source_version)
            break

    config_dir = os.path.expanduser("~/.config/Astra")
    with open(os.path.join(config_dir, 'hmm_databases.json'), 'w') as f:
        json.dump(parsed_json, f, indent=4)

    print(f"{db_name}: {len(expected)} marker profiles installed to {target_folder} "
          f"(KOfam {source_version or 'version unknown'})")

    print(f"\nPressing {db_name} database for fast loading...")
    press_hmm_database(target_folder, db_name=db_name)


def fetch_kofam_markers(kos, target_folder, db_path, rp16):
    """Pull specific KOfam profiles from upstream and inject their thresholds.

    KOfam publishes no per-profile download, so this streams the release
    tarball and keeps only the members we asked for.  Returns the KOfam
    release identifier the profiles came from.
    """
    url = 'https://www.genome.jp/ftp/db/kofam/profiles.tar.gz'
    tar_path = os.path.join(db_path, 'profiles.tar.gz')
    version = remote_version(url)

    if not os.path.exists(tar_path):
        print("Downloading KOfam profiles (no per-profile download is published)...")
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc='profiles.tar.gz') as t:
            urllib.request.urlretrieve(url, tar_path, reporthook=t.update_to)

    wanted = {f"{ko}.hmm": ko for ko in kos}
    found = {}
    with tarfile.open(tar_path, 'r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue
            ko = wanted.get(os.path.basename(member.name))
            if ko is None:
                continue
            dest = os.path.join(target_folder, rp16.profile_filename(ko))
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(dest, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            found[ko] = dest
            if len(found) == len(wanted):
                break

    # Upstream profiles carry no cutoffs; KOfam keeps them in ko_list.
    thresholds = fetch_ko_thresholds(db_path, list(found))
    for ko, dest in found.items():
        if ko in thresholds:
            add_threshold(dest, thresholds[ko])

    return version


def install_databases(db_name, parsed_json=None, db_path=None):
    # Are you trying to install KOFAM? Let's have separate logic for that.
    if db_name == 'KOFAM':
        return install_KOFAM()

    # HydDB also needs special handling for different GA/NC thresholds
    if db_name == 'HydDB':
        return install_HydDB()

    # RP16 is assembled from KOFAM + PFAM rather than downloaded as a unit
    if db_name == 'RP16':
        return install_RP16()

    # Did you call this as a function from an external script?
    # Want to model that function call as 'initialize.install_databases(db_name)'
    # So must leave out parsed_json/db_path
    if parsed_json is None:
        parsed_json = load_json()
    if db_path is None:
        config = load_config()
        db_path = config['db_path']

    # Flag to check if we need to update the JSON file
    update_required = False

    for index, db in enumerate(parsed_json['db_urls']):
        if db['name'] == db_name:
            # Check if database is already installed
            if db['installed'] and db['installation_dir']:
                print(f"Database {db_name} already installed.")
                continue

            # Use the config directory path for database installation
            target_folder = os.path.expandvars(os.path.expanduser(os.path.join(db_path, db_name)))
            print(target_folder)

            if os.path.exists(target_folder) and os.listdir(target_folder):
                print(f"Folder for {db_name} exists and is not empty. Skipping download.")
                if not db.get('installation_dir'):
                    db['installation_dir'] = target_folder
                    update_required = True
                continue

            # Create target directory if it does not exist
            os.makedirs(target_folder, exist_ok=True)
            
            print(f"Downloading {db_name} to {target_folder}...")
            
            # Download the database
            url = db['url']

            if "github.com" in url:
                if '/blob/' in url:
                    # URL points to a single file, convert to raw URL
                    raw_url = url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                    # Proceed to download the single file using the raw URL
                    file_name = raw_url.split('/')[-1]
                    download_path = os.path.join(target_folder, file_name)
                    with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:  
                        urllib.request.urlretrieve(raw_url, download_path, reporthook=t.update_to)
                else:
                    # Special case for GitHub URLs
                    url = url.rstrip("/")
                    if 'Karthik' in db_name:
                        #Hard-code this one because the directory structure is different compared to the other github repos
                        repo_api_url = 'https://api.github.com/repos/kanantharaman/metabolic-hmms/contents/'
                    else:
                        repo_api_url = url.replace("github.com", "api.github.com/repos").replace("/tree/master", "/contents").replace("/tree/main", "/contents")
                    # Paginate through all results (GitHub API caps at 1000 per page by default,
                    # but may return fewer; the Link header signals more pages)
                    all_files = []
                    page_url = repo_api_url + "?per_page=1000"
                    while page_url:
                        response = requests.get(page_url)
                        if response.status_code != 200:
                            print(f"Failed to fetch GitHub directory: {response.status_code}")
                            break
                        all_files.extend(response.json())
                        # Check for next page via Link header
                        page_url = None
                        link_header = response.headers.get("Link", "")
                        for part in link_header.split(","):
                            if 'rel="next"' in part:
                                page_url = part.split(";")[0].strip().strip("<>")
                    for file in all_files:
                        file_name = file['name']
                        if file_name.lower().endswith('.hmm'):  # Only download .hmm or .HMM files
                            file_url = file['download_url']
                            download_path = os.path.join(target_folder, file_name)
                            with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:
                                urllib.request.urlretrieve(file_url, download_path, reporthook=t.update_to)
                    if not all_files:
                        print(f"No files found at {repo_api_url}")


            # Check if the URL points to a directory (ends with '/')
            elif url.endswith('/'):
                subprocess.run(["wget", "-r", "-nH", "--cut-dirs=1", "-P", target_folder, url])
            else:
                file_name = url.split('/')[-1]
                download_path = os.path.join(target_folder, file_name)
                
                # Download the file with progress bar
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=file_name) as t:  
                    urllib.request.urlretrieve(url, download_path, reporthook=t.update_to)
                
                # Extract the file if it's a tar archive
                if tarfile.is_tarfile(download_path):
                    print(f"Extracting {file_name}...")
                    with tarfile.open(download_path, 'r') as tar_ref:
                        tar_ref.extractall(target_folder)
                    os.remove(download_path)  # Remove the original tar file
                    
                    # Check if a single directory was extracted
                    extracted_files = os.listdir(target_folder)
                    if len(extracted_files) == 1 and os.path.isdir(os.path.join(target_folder, extracted_files[0])):
                        single_dir = os.path.join(target_folder, extracted_files[0])
                        for file_to_move in os.listdir(single_dir):
                            shutil.move(os.path.join(single_dir, file_to_move), target_folder)
                        os.rmdir(single_dir)  # Remove the now-empty directory

                # Extract the file if it's a gz archive
                elif file_name.endswith('.gz'):
                    print(f"Decompressing {file_name}...")
                    with gzip.open(download_path, 'rb') as f_in:
                        with open(download_path[:-3], 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    os.remove(download_path)  # Remove the original gz file
                    
                # Check for other file types and add .hmm extension if necessary
                elif not file_name.endswith(('.hmm', '.HMM')):
                    if db_name == 'dbCAN':
                        #dbCAN provides a single file called 'dbCAN-fam-HMMs.txt.v11'
                        new_file_name = file_name + '.hmm'
                        os.rename(download_path, os.path.join(target_folder, new_file_name))
                    else:
                        os.system('rm ' + os.path.join(target_folder, file_name))

            
            print(f"{db_name} successfully downloaded and extracted.")

            # Press the database for fast loading at search time
            print(f"\nPressing {db_name} database for fast loading...")
            press_hmm_database(target_folder, db_name=db_name)

            parsed_json['db_urls'][index]['installed'] = True
            parsed_json['db_urls'][index]['installation_dir'] = target_folder
            record_version(parsed_json['db_urls'][index], url)
            update_required = True
            break  # Break after updating the relevant database


    if update_required:
        # Update the JSON file in the config directory
        config_dir = os.path.expanduser("~/.config/Astra")
        json_path = os.path.join(config_dir, 'hmm_databases.json')
        with open(json_path, 'w') as f:
            json.dump(parsed_json, f, indent=4)



def show_installed_databases(parsed_json):
    print("Installed databases:")
    
    installed_protein_dbs = [db for db in parsed_json['db_urls'] if db['installed'] and db['molecule_type'] == 'protein']
    installed_nucleotide_dbs = [db for db in parsed_json['db_urls'] if db['installed'] and db['molecule_type'] == 'nucleotide']
    
    def describe(db):
        line = f"    - {db['name']} (Installed in: {db['installation_dir']})"
        if db.get('version'):
            line += f"\n        release {db['version']}"
            if db.get('installed_date'):
                line += f", installed {db['installed_date']}"
        return line

    if installed_protein_dbs:
        print("  Protein Databases:")
        for db in installed_protein_dbs:
            print(describe(db))

    if installed_nucleotide_dbs:
        print("  Nucleotide Databases:")
        for db in installed_nucleotide_dbs:
            print(describe(db))



def main(args):

    # Load configuration to get the database path
    parsed_json = initialize_config()

    
    
    db_path = parsed_json['db_path']
    
    # Extract HMM database names from command line arguments
    hmms = args.hmms
    
    # Show available databases if the flag is set
    if args.show_available:
        show_available_databases(parsed_json)
        sys.exit()
        
    # Show installed databases if the flag is set
    if args.show_installed:
        show_installed_databases(parsed_json)
        sys.exit()
        
    # If no HMM database names are provided, show available databases and return
    if hmms is None or len(hmms) == 0:
        show_available_databases(parsed_json)
        return

    # Split the HMM names if multiple databases are provided in a comma-delimited string
    if ',' in hmms:
        hmms = hmms.split(',')
    else:
        hmms = [hmms]

    # Press existing databases if the flag is set
    if getattr(args, 'press', False):
        for db in parsed_json['db_urls']:
            if db['installed'] and db['installation_dir']:
                if hmms and db['name'] not in hmms:
                    continue
                print(f"\nPressing {db['name']}...")
                press_hmm_database(db['installation_dir'], db_name=db['name'])
        return

    # Reinstalling on top of an existing install is a no-op unless we clear the
    # flags first: every installer short-circuits on 'already installed'.
    if getattr(args, 'force', False):
        named = [db for db in parsed_json['db_urls'] if db['name'] in hmms]
        for db in named:
            if db['installed'] and db['installation_dir']:
                stale_dir = os.path.expandvars(os.path.expanduser(db['installation_dir']))
                if os.path.isdir(stale_dir):
                    print(f"--force: removing the existing {db['name']} install at {stale_dir}")
                    shutil.rmtree(stale_dir)
            db['installed'] = False
            db['installation_dir'] = ''

        if named:
            config_dir = os.path.expanduser("~/.config/Astra")
            with open(os.path.join(config_dir, 'hmm_databases.json'), 'w') as f:
                json.dump(parsed_json, f, indent=4)

    # Install the databases
    # Iterate through user-provided database names or special keywords for batch installation
    for db_name in hmms:
        # Case for installing all protein databases
        if db_name == 'all_prot':
            for db in parsed_json['db_urls']:
                if db['molecule_type'] == 'protein' and not db['installed']:
                    install_databases(db['name'], parsed_json, db_path)
                    
        # Case for installing all nucleotide databases
        elif db_name == 'all_nuc':
            for db in parsed_json['db_urls']:
                if db['molecule_type'] == 'nucleotide' and not db['installed']:
                    install_databases(db['name'], parsed_json, db_path)
                    
        # Case for installing a specific database
        else:
            install_databases(db_name, parsed_json, db_path)
