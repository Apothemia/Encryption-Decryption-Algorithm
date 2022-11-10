import sys
import subprocess
from importlib.util import find_spec

if find_spec('numpy') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'numpy', '--disable-pip-version-check'])

import numpy as np


class Crypto:

    @staticmethod
    def decrypt(input_file_path, ip_key, decrypt_key, output_file_path):

        def decrypt_text_arr():
            decrypted_line = ''
            for sixteen_chars in ciphertext_arr:

                # Permute array
                permuted_arr = Crypto.initial_permutation(sixteen_chars, ip_key)

                # Split into two nibbles
                left_nibble, right_nibble = Crypto.split_into_nibbles(permuted_arr)

                # XOR nibbles
                left_nibble_xor = Crypto.xor(left_nibble, key_l1)
                right_nibble_xor = Crypto.xor(right_nibble, key_r1)

                # Swap nibbles
                left_nibble = right_nibble_xor
                right_nibble = left_nibble_xor

                # XOR nibbles again
                left_nibble_xor = Crypto.xor(left_nibble, key_l0)
                right_nibble_xor = Crypto.xor(right_nibble, key_r0)

                # Concatenate nibbles
                concat_arr = np.concatenate([left_nibble_xor, right_nibble_xor], axis=None)

                # Shift right by 8
                shifted_arr = Crypto.shift_right_rotation(concat_arr, by=8)

                # Inverse permute
                inverse_perm_arr = Crypto.inverse_permutation(shifted_arr, ip_key)

                # Decode characters
                decoded_arr = Crypto.decode(inverse_perm_arr)

                # Concatenate array elements
                decrypted_line += ''.join(list(decoded_arr))

            return decrypted_line.replace('\r\n', '\n').strip()

        # Read ciphertext
        ciphertext = Crypto.read_file(filepath=input_file_path)

        # Split ciphertext into lines of 16 characters
        ciphertext_arr = Crypto.split_by_x(text=ciphertext, x=16)

        # Generate XOR keys
        key_l0, key_r0, key_l1, key_r1 = Crypto.generate_keys(decrypt_key)

        # Generate decrypted text
        decrypted_text = decrypt_text_arr()

        Crypto.write_file(output_text=decrypted_text, filepath=output_file_path)
        print(f'DEC> The decrypted text is written into {output_file_path}\n')

    @staticmethod
    def read_file(filepath):
        try:
            with open(filepath, 'rb') as io_file:
                return io_file.read()
        except FileNotFoundError:
            raise SystemExit(f'!!!> Error: No such file: "{filepath}"')

    @staticmethod
    def write_file(output_text, filepath):
        with open(filepath, 'w', encoding='utf8') as io_file:
            io_file.write(output_text)

    @staticmethod
    def encode(input_arr):
        if input_arr.ndim == 1:
            return np.array([ord(x) for x in input_arr])
        elif input_arr.ndim == 2:
            return np.array([[ord(x) for x in line] for line in input_arr])
        else:
            print('The array isn\'t one or two dimensional')
            exit(0)

    @staticmethod
    def decode(input_arr):
        if input_arr.ndim == 1:
            return np.array([chr(x) for x in input_arr])
        elif input_arr.ndim == 2:
            return np.array([[chr(x) for x in line] for line in input_arr])
        else:
            print('The array isn\'t one or two dimensional')
            exit(0)

    @staticmethod
    def split_by_x(x, text, dummy_char=None):
        text_list = list(text)

        if dummy_char is not None:
            while True:
                if len(text_list) % x == 0:
                    break
                text_list.append(ord(dummy_char))

        text_arr = np.array(text_list, dtype=np.int32)
        return text_arr.reshape(-1, x)

    @staticmethod
    def split_into_nibbles(input_arr):
        return input_arr[:8], input_arr[8:]

    @staticmethod
    def generate_keys(decrypt_key):

        encoded_key = Crypto.encode(np.array(list(decrypt_key)))
        encoded_key_arr = Crypto.split_by_x(text=encoded_key, x=4)

        key_l0 = np.concatenate([encoded_key_arr[:, 1], encoded_key_arr[:, 3]], axis=None)
        key_r0 = np.concatenate([encoded_key_arr[:, 0], encoded_key_arr[:, 2]], axis=None)
        key_l1 = np.concatenate([encoded_key_arr[:, 2], encoded_key_arr[:, 3]], axis=None)
        key_r1 = np.concatenate([encoded_key_arr[:, 0], encoded_key_arr[:, 1]], axis=None)

        return key_l0, key_r0, key_l1, key_r1

    @staticmethod
    def initial_permutation(input_arr, perm_key):
        new_list = [input_arr[ip_index] for ip_index in perm_key]
        return np.array(new_list)

    @staticmethod
    def inverse_permutation(input_arr, perm_key):
        inverse_arr = np.array([None] * len(input_arr))
        for char_index, char in zip(perm_key, input_arr):
            inverse_arr[char_index] = char
        return inverse_arr

    @staticmethod
    def shift_right_rotation(input_arr, by=8):
        return np.roll(input_arr, by)

    @staticmethod
    def xor(arr_one, arr_two):
        result = []
        for x, y in zip(arr_one, arr_two):
            result.append(x ^ y)
        return np.array(result)


if __name__ == '__main__':
    dec_key = 'Aleyna tatildemi'
    permutation_key = [9, 7, 1, 6, 5, 15, 4, 2, 8, 10, 11, 13, 14, 3, 0, 12]

    try:
        ciphertext_path = sys.argv[1]
    except IndexError:
        raise SystemExit(f'!!!> Usage: python {sys.argv[0]} [ ciphertext_path ]\n')

    path_split = ciphertext_path.split('.')
    if path_split.pop() != 'enc':
        raise SystemExit('!!!> Error: You need to provide a .enc file to decrypt.\n')

    Crypto.decrypt(input_file_path=ciphertext_path,
                   ip_key=permutation_key,
                   decrypt_key=dec_key,
                   output_file_path='.'.join(path_split) + '.dec')
