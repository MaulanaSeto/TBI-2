import array

class StandardPostings:
    """ 
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """ 
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """ 
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)

class OptPForDeltaPostings:
    """
    Implementasi Optimized Patched Frame-of-Reference Delta (OptPForDelta) encoding.

    OptPForDelta adalah algoritma kompresi bit-level yang:
    1. Menggunakan gap-based encoding (seperti VBE)
    2. Membagi data menjadi blok-blok dengan ukuran tetap
    3. Untuk setiap blok, menghitung jumlah bit minimum (b) yang diperlukan
       untuk merepresentasikan sebagian besar nilai
    4. Nilai-nilai yang tidak muat dalam b bit disimpan sebagai "exceptions"

    Algoritma ini efisien untuk postings list karena gap-gap biasanya kecil
    dan dapat direpresentasikan dengan sedikit bit.
    """

    BLOCK_SIZE = 128  # Ukuran blok standar untuk PForDelta

    @staticmethod
    def _bits_needed(n):
        """Menghitung jumlah bit yang diperlukan untuk merepresentasikan n"""
        if n == 0:
            return 1
        return n.bit_length()

    @staticmethod
    def _find_optimal_b(values, threshold=0.9):
        """
        Mencari nilai b optimal sehingga setidaknya threshold% nilai
        dapat direpresentasikan dengan b bit.

        Parameters
        ----------
        values: List[int]
            List of integers to encode
        threshold: float
            Persentase minimum nilai yang harus muat dalam b bit (default 90%)

        Returns
        -------
        int
            Jumlah bit optimal (b)
        """
        if not values:
            return 1

        # Hitung distribusi bit yang dibutuhkan
        bit_counts = [OptPForDeltaPostings._bits_needed(v) for v in values]
        max_bits = max(bit_counts)

        # Cari b terkecil yang mencakup threshold% nilai
        n = len(values)
        for b in range(1, max_bits + 1):
            count = sum(1 for bc in bit_counts if bc <= b)
            if count / n >= threshold:
                return b

        return max_bits

    @staticmethod
    def _pack_bits(values, num_bits):
        """
        Pack list of integers ke dalam bytearray menggunakan num_bits per nilai.

        Parameters
        ----------
        values: List[int]
            Values to pack (diasumsikan semua muat dalam num_bits)
        num_bits: int
            Number of bits per value

        Returns
        -------
        bytes
            Packed byte representation
        """
        if not values:
            return b''

        # Total bits needed
        total_bits = len(values) * num_bits
        total_bytes = (total_bits + 7) // 8

        result = bytearray(total_bytes)
        bit_pos = 0

        for value in values:
            # Pack value into bit stream
            byte_idx = bit_pos // 8
            bit_offset = bit_pos % 8

            # Handle value across multiple bytes
            remaining_bits = num_bits
            curr_value = value

            while remaining_bits > 0:
                if byte_idx >= len(result):
                    break

                # Bits available in current byte
                available = 8 - bit_offset
                bits_to_write = min(remaining_bits, available)

                # Extract bits to write (from MSB side of remaining value)
                shift = remaining_bits - bits_to_write
                bits = (curr_value >> shift) & ((1 << bits_to_write) - 1)

                # Write to current byte
                result[byte_idx] |= bits << (available - bits_to_write)

                remaining_bits -= bits_to_write
                curr_value &= (1 << shift) - 1 if shift > 0 else 0
                bit_offset = 0
                byte_idx += 1

            bit_pos += num_bits

        return bytes(result)

    @staticmethod
    def _unpack_bits(data, num_bits, count):
        """
        Unpack bytes menjadi list of integers.

        Parameters
        ----------
        data: bytes
            Packed byte data
        num_bits: int
            Number of bits per value
        count: int
            Number of values to unpack

        Returns
        -------
        List[int]
            Unpacked values
        """
        if not data or count == 0:
            return []

        result = []
        bit_pos = 0

        for _ in range(count):
            value = 0
            remaining_bits = num_bits

            while remaining_bits > 0:
                byte_idx = bit_pos // 8
                bit_offset = bit_pos % 8

                if byte_idx >= len(data):
                    break

                # Bits available in current byte
                available = 8 - bit_offset
                bits_to_read = min(remaining_bits, available)

                # Read bits from current byte
                shift = available - bits_to_read
                bits = (data[byte_idx] >> shift) & ((1 << bits_to_read) - 1)

                # Add to value
                value = (value << bits_to_read) | bits

                remaining_bits -= bits_to_read
                bit_pos += bits_to_read

            result.append(value)

        return result

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list dengan OptPForDelta encoding.

        Format:
        - 4 bytes: jumlah total posting
        - Untuk setiap blok:
            - 1 byte: b (jumlah bit per nilai)
            - 2 bytes: jumlah exceptions
            - b-bit packed values (dengan 0 untuk exceptions)
            - Exceptions: posisi (2 bytes) + nilai VBE encoded

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            Encoded postings
        """
        if not postings_list:
            return (0).to_bytes(4, 'little')

        # Convert to gaps (delta encoding)
        gaps = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i-1])

        result = bytearray()

        # Write total count
        result.extend(len(postings_list).to_bytes(4, 'little'))

        # Process in blocks
        BLOCK_SIZE = OptPForDeltaPostings.BLOCK_SIZE

        for block_start in range(0, len(gaps), BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, len(gaps))
            block = gaps[block_start:block_end]
            block_len = len(block)

            # Find optimal b
            b = OptPForDeltaPostings._find_optimal_b(block)
            max_val = (1 << b) - 1

            # Separate normal values and exceptions
            normal_values = []
            exceptions = []  # (position, value)

            for i, val in enumerate(block):
                if val > max_val:
                    normal_values.append(0)  # Placeholder
                    exceptions.append((i, val))
                else:
                    normal_values.append(val)

            # Write block header
            result.append(b)  # 1 byte for b
            result.append(block_len)  # 1 byte for block length
            result.extend(len(exceptions).to_bytes(2, 'little'))  # 2 bytes for exception count

            # Write packed normal values
            packed = OptPForDeltaPostings._pack_bits(normal_values, b)
            result.extend(len(packed).to_bytes(2, 'little'))  # 2 bytes for packed length
            result.extend(packed)

            # Write exceptions
            for pos, val in exceptions:
                result.append(pos)  # Position dalam blok (max 128, muat 1 byte)
                # Encode value dengan VBE
                vbe_encoded = VBEPostings.vb_encode_number(val)
                result.extend(vbe_encoded)

        return bytes(result)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode OptPForDelta encoded postings list.

        Parameters
        ----------
        encoded_postings_list: bytes
            Encoded postings

        Returns
        -------
        List[int]
            Decoded postings list
        """
        if len(encoded_postings_list) < 4:
            return []

        # Read total count
        total_count = int.from_bytes(encoded_postings_list[:4], 'little')
        if total_count == 0:
            return []

        gaps = []
        pos = 4

        while len(gaps) < total_count and pos < len(encoded_postings_list):
            # Read block header
            b = encoded_postings_list[pos]
            pos += 1

            block_len = encoded_postings_list[pos]
            pos += 1

            num_exceptions = int.from_bytes(encoded_postings_list[pos:pos+2], 'little')
            pos += 2

            packed_len = int.from_bytes(encoded_postings_list[pos:pos+2], 'little')
            pos += 2

            # Read packed normal values
            packed_data = encoded_postings_list[pos:pos+packed_len]
            pos += packed_len

            block_values = OptPForDeltaPostings._unpack_bits(packed_data, b, block_len)

            # Read and apply exceptions
            for _ in range(num_exceptions):
                exc_pos = encoded_postings_list[pos]
                pos += 1

                # Decode VBE value
                n = 0
                while pos < len(encoded_postings_list):
                    byte = encoded_postings_list[pos]
                    pos += 1
                    if byte < 128:
                        n = 128 * n + byte
                    else:
                        n = 128 * n + (byte - 128)
                        break

                if exc_pos < len(block_values):
                    block_values[exc_pos] = n

            gaps.extend(block_values)

        # Convert gaps back to postings
        if not gaps:
            return []

        postings = [gaps[0]]
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])

        return postings[:total_count]

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode term frequency list dengan OptPForDelta.
        TF tidak perlu gap encoding karena tidak terurut.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            Encoded TF list
        """
        if not tf_list:
            return (0).to_bytes(4, 'little')

        result = bytearray()

        # Write total count
        result.extend(len(tf_list).to_bytes(4, 'little'))

        # Process in blocks
        BLOCK_SIZE = OptPForDeltaPostings.BLOCK_SIZE

        for block_start in range(0, len(tf_list), BLOCK_SIZE):
            block_end = min(block_start + BLOCK_SIZE, len(tf_list))
            block = tf_list[block_start:block_end]
            block_len = len(block)

            # Find optimal b
            b = OptPForDeltaPostings._find_optimal_b(block)
            max_val = (1 << b) - 1

            # Separate normal values and exceptions
            normal_values = []
            exceptions = []

            for i, val in enumerate(block):
                if val > max_val:
                    normal_values.append(0)
                    exceptions.append((i, val))
                else:
                    normal_values.append(val)

            # Write block header
            result.append(b)
            result.append(block_len)
            result.extend(len(exceptions).to_bytes(2, 'little'))

            # Write packed normal values
            packed = OptPForDeltaPostings._pack_bits(normal_values, b)
            result.extend(len(packed).to_bytes(2, 'little'))
            result.extend(packed)

            # Write exceptions
            for pos, val in exceptions:
                result.append(pos)
                vbe_encoded = VBEPostings.vb_encode_number(val)
                result.extend(vbe_encoded)

        return bytes(result)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode OptPForDelta encoded TF list.

        Parameters
        ----------
        encoded_tf_list: bytes
            Encoded TF list

        Returns
        -------
        List[int]
            Decoded TF list
        """
        if len(encoded_tf_list) < 4:
            return []

        # Read total count
        total_count = int.from_bytes(encoded_tf_list[:4], 'little')
        if total_count == 0:
            return []

        values = []
        pos = 4

        while len(values) < total_count and pos < len(encoded_tf_list):
            # Read block header
            b = encoded_tf_list[pos]
            pos += 1

            block_len = encoded_tf_list[pos]
            pos += 1

            num_exceptions = int.from_bytes(encoded_tf_list[pos:pos+2], 'little')
            pos += 2

            packed_len = int.from_bytes(encoded_tf_list[pos:pos+2], 'little')
            pos += 2

            # Read packed normal values
            packed_data = encoded_tf_list[pos:pos+packed_len]
            pos += packed_len

            block_values = OptPForDeltaPostings._unpack_bits(packed_data, b, block_len)

            # Read and apply exceptions
            for _ in range(num_exceptions):
                exc_pos = encoded_tf_list[pos]
                pos += 1

                # Decode VBE value
                n = 0
                while pos < len(encoded_tf_list):
                    byte = encoded_tf_list[pos]
                    pos += 1
                    if byte < 128:
                        n = 128 * n + byte
                    else:
                        n = 128 * n + (byte - 128)
                        break

                if exc_pos < len(block_values):
                    block_values[exc_pos] = n

            values.extend(block_values)

        return values[:total_count]


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, "hasil decoding tidak sama dengan postings original"
        assert decoded_tf_list == tf_list, "hasil decoding tidak sama dengan postings original"
        print()

    # Test dengan data yang lebih besar
    print("=== Test dengan data besar ===")
    import random
    large_postings = sorted(random.sample(range(1, 100000), 500))
    large_tf = [random.randint(1, 100) for _ in range(500)]

    for Postings in [StandardPostings, VBEPostings, OptPForDeltaPostings]:
        encoded_p = Postings.encode(large_postings)
        encoded_tf = Postings.encode_tf(large_tf)
        decoded_p = Postings.decode(encoded_p)
        decoded_tf = Postings.decode_tf(encoded_tf)

        print(f"{Postings.__name__}:")
        print(f"  Postings: {len(encoded_p)} bytes, TF: {len(encoded_tf)} bytes")
        assert decoded_p == large_postings, f"{Postings.__name__} postings decode failed"
        assert decoded_tf == large_tf, f"{Postings.__name__} TF decode failed"
    print("Semua test berhasil!")
