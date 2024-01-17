import util
import re
import sys
# -*- coding:utf-8 -*-

def make_kmer_list(k, alphabet):
    """
    Generate kmer list.
    """
    if k < 0:
        print("Error, k must be an inter and larger than 0.")

    kmers = []
    for i in range(1, k + 1):
        if len(kmers) == 0:
            kmers = list(alphabet)
        else:
            new_kmers = []
            for kmer in kmers:
                for c in alphabet:
                    new_kmers.append(kmer + c)
            kmers = new_kmers

    return kmers

def find_revcomp(sequence, revcomp_dictionary):
    # Save time by storing reverse complements in a hash.
    if sequence in revcomp_dictionary:
        return revcomp_dictionary[sequence]

    # Make a reversed version of the string.
    rev_sequence = list(sequence)
    rev_sequence.reverse()
    rev_sequence = ''.join(rev_sequence)

    return_value = ""
    for letter in rev_sequence:
        if letter == "A":
            return_value += "T"
        elif letter == "C":
            return_value += "G"
        elif letter == "G":
            return_value += "C"
        elif letter == "T":
            return_value += "A"
        elif letter == "N":
            return_value += "N"
        else:
            error_info = ("Unknown DNA character (%s)\n" % letter)
            sys.exit(error_info)

    # Store this value for future use.
    revcomp_dictionary[sequence] = return_value

    return return_value

def _cmp(a, b):
    return (a > b) - (a < b)

def make_revcomp_kmer_list(kmer_list):
    revcomp_dictionary = {}
    new_kmer_list = [kmer for kmer in kmer_list if _cmp(kmer, find_revcomp(kmer, revcomp_dictionary)) <= 0]
    return new_kmer_list


def make_kmer_vector(k, seq, revcomp=False):
    alphabet = 'ACGU'

    """Generate kmer vector."""

    if revcomp and re.search(r'[^acgtACGT]', ''.join(alphabet)) is not None:
        sys.exit("Error, Only DNA sequence can be reverse compliment.")

    kmer_list = make_kmer_list(k, alphabet)
    count_sum = 0

    # Generate the kmer frequency dict.
    kmer_count = {}
    for kmer in kmer_list:
        temp_count = util.frequency(seq, kmer)
        if not revcomp:
            if kmer not in kmer_count:
                kmer_count[kmer] = 0
            kmer_count[kmer] += temp_count
        else:
            rev_kmer = find_revcomp(kmer, {})
            if kmer <= rev_kmer:
                if kmer not in kmer_count:
                    kmer_count[kmer] = 0
                kmer_count[kmer] += temp_count
            else:
                if rev_kmer not in kmer_count:
                    kmer_count[rev_kmer] = 0
                kmer_count[rev_kmer] += temp_count

        count_sum += temp_count

    # Normalize.
    if not revcomp:
        count_vec = [kmer_count[kmer] for kmer in kmer_list]
    else:
        revc_kmer_list = make_revcomp_kmer_list(kmer_list)
        count_vec = [kmer_count[kmer] for kmer in revc_kmer_list]
    count_vec = [round(float(e) / count_sum, 8) for e in count_vec]

    return count_vec