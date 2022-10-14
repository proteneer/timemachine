class TestGRPCClient(unittest.TestCase):
    def setUp(self):

        starting_port = random.randint(2000, 5000)
        # setup server, in reality max_workers is equal to number of gpus
        self.ports = [starting_port + i for i in range(2)]
        self.servers = []
        for port in self.ports:
            server = grpc.server(
                concurrent.futures.ThreadPoolExecutor(max_workers=1),
                options=[
                    ("grpc.max_send_message_length", 50 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 50 * 1024 * 1024),
                ],
            )
            parallel.grpc.service_pb2_grpc.add_WorkerServicer_to_server(worker.Worker(), server)
            server.add_insecure_port("[::]:" + str(port))
            server.start()
            self.servers.append(server)

        # setup client
        self.hosts = [f"0.0.0.0:{port}" for port in self.ports]
        self.cli = client.GRPCClient(self.hosts)

    @patch("timemachine.parallel.worker.get_worker_status")
    def test_checking_host_status(self, mock_status):
        assert self.cli.max_workers == 2
        # All the workers return the same thing
        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver="foo", git_sha="bar") for _ in self.servers
        ]
        self.cli.verify()

        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver=f"foo{i}", git_sha=f"bar{i}")
            for i in range(len(self.servers))
        ]

        with self.assertRaises(AssertionError):
            self.cli.verify()

    @patch("timemachine.parallel.worker.get_worker_status")
    def test_unavailable_host(self, mock_status):

        hosts = self.hosts.copy()
        bad_host = "128.128.128.128:8888"
        hosts.append(bad_host)  # Give it a bad connexion, should fail
        cli = client.GRPCClient(hosts)

        # All the workers return the same thing
        mock_status.side_effect = [
            parallel.grpc.service_pb2.StatusResponse(nvidia_driver="foo", git_sha="bar") for _ in self.servers
        ]

        with self.assertRaises(AssertionError) as e:
            cli.verify()
        self.assertIn(bad_host, str(e.exception))

    def test_default_port(self):
        host = "128.128.128.128"
        cli = client.GRPCClient([host], default_port=9999)
        self.assertEqual(cli.hosts[0], "128.128.128.128:9999")

    def test_hosts_from_file(self):
        with self.assertRaises(AssertionError):
            cli = client.GRPCClient("nosuchfile", default_port=9999)

        hosts = ["128.128.128.128", "127.127.127.127:8888"]
        with NamedTemporaryFile(suffix=".txt") as temp:
            for host in hosts:
                temp.write(f"{host}\n".encode("utf-8"))
            temp.flush()
            cli = client.GRPCClient(temp.name, default_port=9999)
            self.assertEqual(cli.hosts[0], "128.128.128.128:9999")
            self.assertEqual(cli.hosts[1], hosts[1])

    def test_foo_2_args(self):
        xs = np.linspace(0, 1.0, 5)
        ys = np.linspace(1.2, 2.2, 5)

        futures = []
        for x, y in zip(xs, ys):
            fut = self.cli.submit(mult, x, y)
            futures.append(fut)

        test_res = []
        test_ids = []
        test_names = []
        for f in futures:
            test_res.append(f.result())
            test_ids.append(f.id)
            test_names.append(f.name)

        expected_ids = ["0", "1", "2", "3", "4"]
        assert test_ids == expected_ids
        assert test_names == expected_ids

        np.testing.assert_array_equal(test_res, xs * ys)

    def test_foo_1_arg(self):
        xs = np.linspace(0, 1.0, 5)

        futures = []
        for x in xs:
            fut = self.cli.submit(square, x)
            futures.append(fut)

        test_res = []
        test_ids = []
        test_names = []
        for f in futures:
            test_res.append(f.result())
            test_ids.append(f.id)
            test_names.append(f.name)

        expected_ids = ["0", "1", "2", "3", "4"]
        assert test_ids == expected_ids
        assert test_names == expected_ids

        np.testing.assert_array_equal(test_res, xs * xs)

    def tearDown(self):
        for server in self.servers:
            server.stop(5)